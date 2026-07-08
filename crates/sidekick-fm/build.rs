//! Compiles the Swift Foundation Models shim on macOS hosts.
//!
//! Anywhere the shim can't be built by design (non-macOS host/target, or
//! `SIDEKICK_FM_STUB=1`), we emit `cfg(fm_stub)` and the crate compiles a
//! stub backend that reports `Unavailable`. This keeps Linux CI and
//! cross-checks working while real behavior lights up on a Mac with the
//! macOS 26 SDK.
//!
//! On macOS the shim is mandatory: a missing or failing Swift toolchain is a
//! hard build error, not a silent stub fallback. A daemon that builds green
//! but answers every chat request with 503 `NotSupportedInBuild` is much
//! harder to diagnose than a build failure with a hint. Opt into the stub
//! explicitly with `SIDEKICK_FM_STUB=1` if that's really what you want.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(fm_stub)");
    // Watch the specific Swift source file, never a directory that could
    // also receive build output — a dirty output dir forces a full Swift
    // recompile on every cargo invocation.
    println!("cargo::rerun-if-changed=swift/bridge.swift");
    println!("cargo::rerun-if-env-changed=SIDEKICK_FM_STUB");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let host_is_macos = cfg!(target_os = "macos");
    let forced_stub = std::env::var("SIDEKICK_FM_STUB").map(|v| v == "1").unwrap_or(false);

    if target_os != "macos" || !host_is_macos || forced_stub {
        println!("cargo::rustc-cfg=fm_stub");
        return;
    }

    // Preflight: fail with an actionable message if the Swift toolchain is
    // missing, rather than a cryptic spawn error from the compile below.
    let swiftc_ok = Command::new("xcrun")
        .args(["swiftc", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !swiftc_ok {
        panic!(
            "sidekick-fm: `xcrun swiftc` is not available. Install the Xcode 26 \
             command line tools (`xcode-select --install`) or select a full Xcode \
             (`sudo xcode-select -s /Applications/Xcode.app`). To build the stub \
             backend instead, set SIDEKICK_FM_STUB=1."
        );
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_path = out_dir.join("libsidekick_fm_bridge.a");

    // Build a static library from the Swift shim. -swift-version 5 keeps
    // strict-concurrency diagnostics from rejecting the semaphore bridging.
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-emit-library",
            "-static",
            "-swift-version",
            "5",
            "-O",
            "-module-name",
            "sidekick_fm_bridge",
            "-target",
            &format!(
                "{}-apple-macosx26.0",
                std::env::var("CARGO_CFG_TARGET_ARCH").unwrap()
            ),
            "swift/bridge.swift",
            "-o",
            lib_path.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo::rustc-link-search=native={}", out_dir.display());
            println!("cargo::rustc-link-lib=static=sidekick_fm_bridge");
            // Swift runtime + frameworks the shim needs.
            println!("cargo::rustc-link-search=native=/usr/lib/swift");
            println!("cargo::rustc-link-lib=framework=Foundation");
            println!("cargo::rustc-link-lib=framework=FoundationModels");
            // The static shim references Swift runtime dylibs by @rpath, so
            // every binary linking it needs an rpath to the system Swift
            // runtime or it aborts at dyld load time. This covers this
            // crate's own test binaries; downstream binary crates must emit
            // the same flag from their build.rs (see sidekick-server).
            println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
            if let Ok(output) = Command::new("xcrun")
                .args(["--show-sdk-path"])
                .output()
            {
                let sdk = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo::rustc-link-search=native={sdk}/usr/lib/swift");
            }
        }
        _ => {
            panic!(
                "sidekick-fm: swiftc failed to compile swift/bridge.swift. \
                 Check that Xcode 26+ is selected (`xcodebuild -version`). \
                 To build the stub backend instead, set SIDEKICK_FM_STUB=1."
            );
        }
    }
}
