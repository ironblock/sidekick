//! Compiles the Swift Foundation Models shim on macOS hosts.
//!
//! Anywhere the shim can't be built by design (non-macOS host/target, or
//! `SIDEKICK_FM_STUB=1`), we emit `cfg(fm_stub)` and the crate compiles a
//! stub backend that reports `Unavailable`. This keeps Linux CI and
//! cross-checks working while real behavior lights up on a Mac.
//!
//! On macOS the shim is mandatory: a missing or failing Swift toolchain is a
//! hard build error, not a silent stub fallback. A daemon that builds green
//! but answers every chat request with 503 `NotSupportedInBuild` is much
//! harder to diagnose than a build failure with a hint. Opt into the stub
//! explicitly with `SIDEKICK_FM_STUB=1` if that's really what you want.
//!
//! SDK awareness: the shim's runtime floor is macOS 26.0 (`-target
//! …-apple-macosx26.0`), but which APIs it can *compile* depends on the SDK.
//! The macOS 27 SDK adds symbols (model variant and capabilities, per-response
//! token usage, typed `LanguageModelError`s) that don't exist in 26.x SDKs, so
//! they sit behind `#if SK_SDK_27` in the Swift source, and behind
//! `#available(macOS 27, *)` at runtime. The SDK version is read from the
//! SDK's own `SDKSettings.json` and exported as `SIDEKICK_FM_SDK` so the
//! daemon can report what it was built against. SDKs older than 26.4 are
//! rejected: they lack `tokenCount(for:)` and the back-deployed
//! `contextSize`, and nothing tests that configuration.
//!
//! Switching Xcode with `xcode-select` changes no environment variable, so
//! Cargo can't notice it; run `cargo clean -p sidekick-fm` afterwards. The
//! daemon's `--version` and `/health` show which SDK a binary was built with.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Oldest SDK the shim compiles against.
const MIN_SDK: (u32, u32) = (26, 4);

fn main() {
    println!("cargo::rustc-check-cfg=cfg(fm_stub)");
    // Watch the specific Swift source file, never a directory that could
    // also receive build output — a dirty output dir forces a full Swift
    // recompile on every cargo invocation.
    println!("cargo::rerun-if-changed=swift/bridge.swift");
    println!("cargo::rerun-if-env-changed=SIDEKICK_FM_STUB");
    println!("cargo::rerun-if-env-changed=SDKROOT");
    println!("cargo::rerun-if-env-changed=DEVELOPER_DIR");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let host_is_macos = cfg!(target_os = "macos");
    let forced_stub = std::env::var("SIDEKICK_FM_STUB").map(|v| v == "1").unwrap_or(false);

    if target_os != "macos" || !host_is_macos || forced_stub {
        println!("cargo::rustc-cfg=fm_stub");
        println!("cargo::rustc-env=SIDEKICK_FM_SDK=none");
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
            "sidekick-fm: `xcrun swiftc` is not available. Install Xcode 26.4 or \
             later (Xcode 27 recommended) or its command line tools, and select it \
             with `sudo xcode-select -s /Applications/Xcode.app`. To build the stub \
             backend instead, set SIDEKICK_FM_STUB=1."
        );
    }

    // Resolve one SDK and use it for both the version check and the
    // compile, so they can't disagree. `xcrun --show-sdk-path` honors
    // SDKROOT, then the selected developer directory.
    let sdk = xcrun_sdk_path();
    let sdk_settings = sdk.join("SDKSettings.json");
    println!("cargo::rerun-if-changed={}", sdk_settings.display());
    let (version_text, version) = sdk_version(&sdk_settings);
    if version < MIN_SDK {
        panic!(
            "sidekick-fm: the macOS {version_text} SDK at {} is too old; the \
             Foundation Models shim needs the macOS {}.{} SDK or newer (Xcode {}.{}+, \
             Xcode 27 recommended). Select a newer Xcode with `sudo xcode-select -s`, \
             or set SIDEKICK_FM_STUB=1 to build the stub backend.",
            sdk.display(),
            MIN_SDK.0,
            MIN_SDK.1,
            MIN_SDK.0,
            MIN_SDK.1,
        );
    }
    println!("cargo::rustc-env=SIDEKICK_FM_SDK={version_text}");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_path = out_dir.join("libsidekick_fm_bridge.a");
    let target = format!(
        "{}-apple-macosx26.0",
        std::env::var("CARGO_CFG_TARGET_ARCH").unwrap()
    );

    // Build a static library from the Swift shim. -swift-version 5 keeps
    // strict-concurrency diagnostics from rejecting the semaphore bridging.
    let mut swiftc = Command::new("xcrun");
    swiftc.args(["swiftc", "-emit-library", "-static", "-swift-version", "5", "-O"]);
    swiftc.args(["-module-name", "sidekick_fm_bridge", "-target", &target]);
    swiftc.arg("-sdk").arg(&sdk);
    if version >= (27, 0) {
        swiftc.args(["-D", "SK_SDK_27"]);
    }
    swiftc.arg("swift/bridge.swift").arg("-o").arg(&lib_path);

    match swiftc.status() {
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
            println!("cargo::rustc-link-search=native={}/usr/lib/swift", sdk.display());
        }
        _ => {
            panic!(
                "sidekick-fm: swiftc failed to compile swift/bridge.swift against the \
                 macOS {version_text} SDK at {}. Check the selected Xcode \
                 (`xcodebuild -version`). To build the stub backend instead, set \
                 SIDEKICK_FM_STUB=1.",
                sdk.display()
            );
        }
    }
}

fn xcrun_sdk_path() -> PathBuf {
    let output = Command::new("xcrun")
        .arg("--show-sdk-path")
        .output()
        .unwrap_or_else(|e| panic!("sidekick-fm: `xcrun --show-sdk-path` failed to run: {e}"));
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || path.is_empty() {
        panic!(
            "sidekick-fm: `xcrun --show-sdk-path` found no macOS SDK: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    PathBuf::from(path)
}

/// The SDK's version string and numeric (major, minor), from SDKSettings.json.
/// Read from the file rather than `xcrun --show-sdk-version`, which can't
/// report SDKs outside the selected developer directory.
fn sdk_version(settings: &Path) -> (String, (u32, u32)) {
    let text = std::fs::read_to_string(settings).unwrap_or_else(|e| {
        panic!("sidekick-fm: can't read {}: {e}", settings.display())
    });
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!("sidekick-fm: {} is not valid JSON: {e}", settings.display())
    });
    let version = json
        .get("Version")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("sidekick-fm: no \"Version\" in {}", settings.display()))
        .to_string();
    let mut parts = version.split('.').map(|p| p.parse::<u32>());
    let parsed = match (parts.next(), parts.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => (major, minor),
        (Some(Ok(major)), None) => (major, 0),
        _ => panic!("sidekick-fm: unrecognized SDK version {version:?} in {}", settings.display()),
    };
    (version, parsed)
}
