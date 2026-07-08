//! The sidekick-fm Swift shim references Swift runtime dylibs by @rpath;
//! every binary that links it (sidekickd, test binaries, smoke-test) needs
//! an rpath to the system Swift runtime or dyld aborts at load time.
//! `cargo::rustc-link-arg` from a dependency's build script does not
//! propagate to downstream binaries, so we emit it here too.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
