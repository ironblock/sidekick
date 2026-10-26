//! The `kv-state-1` program contract (docs/design/open-llm.md, "The program
//! contract"), as a check of what a chunk program declares against the
//! shapes its manifest's `[decoder]` table names.
//!
//! Platform-neutral, so the rules are tested everywhere; the decoder reads
//! the declarations from Core ML.

use sidekick_core::chat_manifest::{kv_names, DecoderSection};
use std::collections::BTreeMap;
use std::fmt;

/// A multi-array's element type, as a program declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F16,
    F32,
    F64,
    I32,
    Other,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dtype::F16 => "fp16",
            Dtype::F32 => "fp32",
            Dtype::F64 => "fp64",
            Dtype::I32 => "int32",
            Dtype::Other => "another type",
        })
    }
}

/// A declared input, output or state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDecl {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl TensorDecl {
    pub fn new(shape: &[usize], dtype: Dtype) -> Self {
        Self {
            shape: shape.to_vec(),
            dtype,
        }
    }
}

impl fmt::Display for TensorDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} {}", self.shape, self.dtype)
    }
}

/// What a chunk program declares, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramDecl {
    pub inputs: BTreeMap<String, TensorDecl>,
    pub outputs: BTreeMap<String, TensorDecl>,
    pub states: BTreeMap<String, TensorDecl>,
}

/// The shapes a decoder's programs share, from its manifest's `[decoder]`
/// table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderShape {
    /// T: tokens per call.
    pub tokens_per_call: usize,
    /// C: the KV cache's capacity.
    pub cache_capacity: usize,
    /// C − T + 1: the usable context, prompt plus completion.
    pub context_length: usize,
    /// D.
    pub hidden_size: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    /// Layers per chunk, in chain order.
    pub layers: Vec<usize>,
    /// V.
    pub vocab_size: usize,
}

impl DecoderShape {
    pub fn from_section(d: &DecoderSection) -> Self {
        Self {
            tokens_per_call: d.tokens_per_call,
            cache_capacity: d.cache_capacity,
            context_length: d.context_length,
            hidden_size: d.hidden_size,
            head_dim: d.head_dim,
            kv_heads: d.kv_heads,
            layers: d.layers.clone(),
            vocab_size: d.vocab_size,
        }
    }

    pub fn chunks(&self) -> usize {
        self.layers.len()
    }

    /// What chunk `index` must declare. Every name is the contract's but the
    /// residual stream's input, `hidden_in`, which a build made before the
    /// contract named it may call otherwise ([`ProgramNames`]).
    pub fn expected(&self, index: usize, names: &ProgramNames) -> ProgramDecl {
        let (t, c, d, hd) = (
            self.tokens_per_call,
            self.cache_capacity,
            self.hidden_size,
            self.head_dim,
        );
        let last = index + 1 == self.chunks();
        let mut p = ProgramDecl::default();
        let f16 = |shape: &[usize]| TensorDecl::new(shape, Dtype::F16);
        p.inputs.insert(names.hidden_in.clone(), f16(&[1, d, 1, t]));
        p.inputs
            .insert(kv_names::POS.into(), TensorDecl::new(&[1], Dtype::I32));
        p.inputs.insert(kv_names::MASK.into(), f16(&[1, 1, t, c]));
        p.inputs.insert(kv_names::COS.into(), f16(&[t, hd]));
        p.inputs.insert(kv_names::SIN.into(), f16(&[t, hd]));
        if last {
            p.inputs.insert(kv_names::SEL.into(), f16(&[t]));
            p.outputs
                .insert(kv_names::LOGITS.into(), f16(&[1, self.vocab_size]));
        } else {
            p.outputs
                .insert(kv_names::HIDDEN_OUT.into(), f16(&[1, d, 1, t]));
        }
        let cache = f16(&[self.layers[index], self.kv_heads, c, hd]);
        p.states.insert(kv_names::K_CACHE.into(), cache.clone());
        p.states.insert(kv_names::V_CACHE.into(), cache);
        p
    }
}

/// The input name the residual stream enters by. The contract's is
/// `hidden_in`; #173's spike builds, which predate it, call it `hidden`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramNames {
    pub hidden_in: String,
}

impl Default for ProgramNames {
    fn default() -> Self {
        Self {
            hidden_in: kv_names::HIDDEN_IN.into(),
        }
    }
}

/// Check chunk `index`'s declarations against the contract at `shape`:
/// every input, output and state it must have, at its shape and type, and
/// nothing else. `Err` is a sentence for a skip reason.
pub fn check_program(
    shape: &DecoderShape,
    index: usize,
    names: &ProgramNames,
    got: &ProgramDecl,
) -> Result<(), String> {
    let want = shape.expected(index, names);
    for (kind, want, got) in [
        ("input", &want.inputs, &got.inputs),
        ("output", &want.outputs, &got.outputs),
        ("state", &want.states, &got.states),
    ] {
        for (name, w) in want {
            match got.get(name) {
                None => {
                    return Err(format!(
                        "it has no {kind} `{name}` ({w} in the `kv-state-1` contract)"
                    ))
                }
                Some(g) if g != w => return Err(format!(
                    "its {kind} `{name}` is {g}, and the manifest's `[decoder]` table makes it {w}"
                )),
                Some(_) => {}
            }
        }
        if let Some(extra) = got.keys().find(|k| !want.contains_key(*k)) {
            return Err(format!(
                "it has the {kind} `{extra}`, which the `kv-state-1` contract doesn't"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> DecoderShape {
        DecoderShape {
            tokens_per_call: 16,
            cache_capacity: 2048,
            context_length: 2033,
            hidden_size: 2048,
            head_dim: 128,
            kv_heads: 2,
            layers: [14, 14, 10, 4].into(),
            vocab_size: 130560,
        }
    }

    #[test]
    fn a_conforming_chain_passes() {
        let s = shape();
        let names = ProgramNames::default();
        for i in 0..4 {
            assert_eq!(check_program(&s, i, &names, &s.expected(i, &names)), Ok(()));
        }
        assert!(s.expected(3, &names).inputs.contains_key("sel"));
        assert!(!s.expected(2, &names).inputs.contains_key("sel"));
        assert_eq!(
            s.expected(1, &names).states["k_cache"].shape,
            [14, 2, 2048, 128]
        );
    }

    #[test]
    fn a_wrong_state_shape_is_named() {
        let s = shape();
        let names = ProgramNames::default();
        let mut got = s.expected(2, &names);
        got.states.get_mut("v_cache").unwrap().shape[0] = 11;
        let err = check_program(&s, 2, &names, &got).unwrap_err();
        assert!(
            err.contains("state `v_cache` is [11, 2, 2048, 128] fp16")
                && err.contains("[10, 2, 2048, 128]"),
            "{err}"
        );
    }

    #[test]
    fn missing_extra_and_mistyped_declarations_are_refused() {
        let s = shape();
        let names = ProgramNames::default();
        let mut got = s.expected(3, &names);
        got.inputs.remove("sel");
        assert!(check_program(&s, 3, &names, &got)
            .unwrap_err()
            .contains("no input `sel`"));

        let mut got = s.expected(0, &names);
        got.inputs
            .insert("sel".into(), TensorDecl::new(&[16], Dtype::F16));
        assert!(check_program(&s, 0, &names, &got)
            .unwrap_err()
            .contains("input `sel`"));

        let mut got = s.expected(0, &names);
        got.inputs.get_mut("pos").unwrap().dtype = Dtype::F32;
        let err = check_program(&s, 0, &names, &got).unwrap_err();
        assert!(
            err.contains("`pos` is [1] fp32") && err.contains("[1] int32"),
            "{err}"
        );

        let mut got = s.expected(3, &names);
        got.outputs.get_mut("logits").unwrap().shape = vec![16, 130560];
        assert!(check_program(&s, 3, &names, &got)
            .unwrap_err()
            .contains("output `logits`"));

        let mut got = s.expected(1, &names);
        got.states
            .insert("extra_cache".into(), TensorDecl::new(&[1], Dtype::F16));
        assert!(check_program(&s, 1, &names, &got)
            .unwrap_err()
            .contains("state `extra_cache`"));
    }

    #[test]
    fn the_residual_input_can_carry_a_pre_contract_name() {
        let s = shape();
        let spike = ProgramNames {
            hidden_in: "hidden".into(),
        };
        let got = s.expected(0, &spike);
        assert_eq!(check_program(&s, 0, &spike, &got), Ok(()));
        let err = check_program(&s, 0, &ProgramNames::default(), &got).unwrap_err();
        assert!(err.contains("no input `hidden_in`"), "{err}");
    }
}
