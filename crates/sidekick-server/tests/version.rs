//! `sidekickd --version` names the ONNX Runtime it expects and the model
//! backends compiled in.

#[test]
fn version_lists_the_backends_compiled_in() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sidekickd")).arg("--version").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let version = String::from_utf8(out.stdout).unwrap();
    let backends = sidekick_embed::compiled_backends().join(", ");
    assert!(version.contains(&format!("(backends: {backends})")), "{version}");
    assert!(backends.ends_with("static"), "static is always compiled in: {backends}");
    // tools/smoke_launch.sh reads the expected runtime from here.
    let onnx = format!("(ONNX Runtime {} expected)", sidekick_embed::onnx::PINNED_VERSION);
    assert_eq!(version.contains(&onnx), sidekick_embed::onnx::SUPPORTED, "{version}");
}
