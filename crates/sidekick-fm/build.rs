//! Compiles the Swift Foundation Models shim on macOS hosts.
//!
//! Anywhere the shim can't be built (non-macOS host/target, no Xcode, or
//! `SIDEKICK_FM_STUB=1`), we emit `cfg(fm_stub)` and the crate compiles a
//! stub backend that reports `Unavailable`. This keeps Linux CI and
//! cross-checks working while real behavior lights up on a Mac with the
//! macOS 26 SDK.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(fm_stub)");
    println!("cargo::rerun-if-changed=swift/bridge.swift");
    println!("cargo::rerun-if-env-changed=SIDEKICK_FM_STUB");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let host_is_macos = cfg!(target_os = "macos");
    let forced_stub = std::env::var("SIDEKICK_FM_STUB").map(|v| v == "1").unwrap_or(false);

    if target_os != "macos" || !host_is_macos || forced_stub {
        println!("cargo::rustc-cfg=fm_stub");
        return;
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
            if let Ok(output) = Command::new("xcrun")
                .args(["--show-sdk-path"])
                .output()
            {
                let sdk = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo::rustc-link-search=native={sdk}/usr/lib/swift");
            }
        }
        _ => {
            println!(
                "cargo::warning=sidekick-fm: swiftc unavailable or shim failed to build; \
                 building stub backend (set up Xcode 26 for the real one)"
            );
            println!("cargo::rustc-cfg=fm_stub");
        }
    }
}
