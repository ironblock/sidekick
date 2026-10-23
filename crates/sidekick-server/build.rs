//! The sidekick-fm Swift shim references Swift runtime dylibs by @rpath;
//! every binary that links it (sidekickd, test binaries, smoke-test) needs
//! an rpath to the system Swift runtime or dyld aborts at load time.
//! `cargo::rustc-link-arg` from a dependency's build script does not
//! propagate to downstream binaries, so we emit it here too.
//!
//! It also builds sidekick's checksum ledger (docs/data/ledger.json, #136)
//! into sidekickd, for `sidekickd fetch` to verify downloads against: the
//! hashes a release trusts ship inside it, not beside the files they check.
//! A checkout without the ledger builds with an empty one.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
    let ledger = std::path::Path::new("../../docs/data/ledger.json");
    // The directory too, so the ledger's arrival is noticed.
    println!("cargo::rerun-if-changed=../../docs/data");
    if ledger.exists() {
        println!("cargo::rerun-if-changed={}", ledger.display());
    }
    let text = std::fs::read_to_string(ledger).unwrap_or_else(|_| "{\"schema\": 1, \"builds\": []}".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("ledger.json");
    if std::fs::read_to_string(&out).ok().as_deref() != Some(text.as_str()) {
        std::fs::write(out, text).unwrap();
    }
}
