//! A chunked bucket's programs, checked together at load (D37), for the
//! classifier and the embedder alike.

use sidekick_core::{Error, Result};
use sidekick_coreml::{ModelInterface, ShapeVerdict, HIDDEN_IN, HIDDEN_OUT};
use std::path::PathBuf;

/// What a model's chain must take: the inputs its manifest names (each with
/// its shape where it is fixed), under the manifest table that names them.
pub(crate) struct ChainInputs<'a> {
    pub id: &'a str,
    pub bucket: usize,
    pub inputs: Vec<(&'a String, Option<Vec<usize>>)>,
    pub input_ids: &'a str,
    /// The manifest table naming the inputs, for messages: `[io]`, `[classify.io]`.
    pub table: &'a str,
}

/// Check a chunked bucket's programs, named relative to the model
/// directory: together they take every input the manifest names (each at
/// its shape), the first takes `input_ids` and no `hidden_in`, every later
/// one takes `hidden_in`, which the one before outputs as `hidden_out` at
/// the same [1, bucket, H] shape, no chunk takes an input the manifest
/// doesn't name, and the last produces the manifest's output, as
/// `check_output` judges it (given the last program's interface and a
/// failure that names that program).
pub(crate) fn check_chain(
    want: &ChainInputs<'_>,
    names: &[String],
    paths: &[PathBuf],
    check_output: impl FnOnce(&ModelInterface, &dyn Fn(String) -> Result<()>) -> Result<()>,
) -> Result<()> {
    let bucket = want.bucket;
    let fail_in = |name: &str, message: String| Err(Error::Inference(format!("model `{}`, {name}: {message}", want.id)));
    let mut ifaces = Vec::with_capacity(paths.len());
    for (name, path) in names.iter().zip(paths) {
        let iface = match sidekick_coreml::interface(path) {
            Ok(iface) => iface,
            Err(e) => return fail_in(name, format!("can't read the artifact: {e}")),
        };
        if let ShapeVerdict::Refuse(reason) = sidekick_coreml::load_verdict(&iface.constraints) {
            return fail_in(name, reason);
        }
        ifaces.push(iface);
    }
    let named: Vec<&String> = want.inputs.iter().map(|(n, _)| *n).collect();
    for (i, (name, iface)) in names.iter().zip(&ifaces).enumerate() {
        if let Some(extra) = iface.inputs.keys().find(|k| *k != HIDDEN_IN && !named.contains(k)) {
            return fail_in(name, format!("takes input `{extra}`, which the manifest's {} doesn't name", want.table));
        }
        for (input, shape) in &want.inputs {
            match (iface.inputs.get(*input), shape) {
                (Some(got), Some(shape)) if got != shape => {
                    return fail_in(name, format!("input `{input}` is {got:?}, expected {shape:?}"))
                }
                _ => {}
            }
        }
        let takes_hidden = iface.inputs.get(HIDDEN_IN);
        if i == 0 {
            if takes_hidden.is_some() {
                return fail_in(name, format!("the first chunk takes `{HIDDEN_IN}`; nothing comes before it"));
            }
            if !iface.inputs.contains_key(want.input_ids) {
                return fail_in(name, format!("the first chunk doesn't take `{}`", want.input_ids));
            }
        } else {
            let produced = ifaces[i - 1].outputs.get(HIDDEN_OUT);
            match (produced, takes_hidden) {
                (_, None) => return fail_in(name, format!("a chunk after the first must take `{HIDDEN_IN}`")),
                (None, _) => return fail_in(&names[i - 1], format!("a chunk before the last must output `{HIDDEN_OUT}`")),
                (Some(out), Some(inp)) if out != inp || out.len() != 3 || out[..2] != [1, bucket] => {
                    return fail_in(name, format!(
                        "takes `{HIDDEN_IN}` {inp:?}, and the chunk before outputs `{HIDDEN_OUT}` {out:?}; both must \
                         be the same [1, {bucket}, hidden]"
                    ))
                }
                _ => {}
            }
        }
    }
    for (input, _) in &want.inputs {
        if !ifaces.iter().any(|f| f.inputs.contains_key(*input)) {
            return fail_in(&names[0], format!("no chunk takes the int32 multi-array input `{input}`"));
        }
    }
    let last = names.last().expect("a chain has a program");
    check_output(ifaces.last().expect("a chain has a program"), &|message| fail_in(last, message))
}
