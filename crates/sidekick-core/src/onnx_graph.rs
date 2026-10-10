//! An ONNX model's declared inputs and outputs, read from its protobuf
//! without loading its weights or any runtime: what the registry checks an
//! `onnx` manifest's `[io]` against when it scans (D40).
//!
//! It reads only the fields it needs (`ModelProto.graph`, then the graph's
//! `input`, `output` and `initializer` names) and seeks past everything
//! else, so a large model's inline weights cost a seek, not a read. Field
//! numbers are onnx.proto's.

use std::collections::HashSet;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// ONNX's `TensorProto.DataType` values the checks name.
pub const ONNX_FLOAT: i32 = 1;
pub const ONNX_INT32: i32 = 6;
pub const ONNX_INT64: i32 = 7;

/// One dimension of a declared shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    /// Named, dynamic (`batch`, `sequence`).
    Symbolic(String),
    /// Declared without a value or a name.
    Unknown,
}

/// A declared tensor: its name, element type, and shape (`None` when the
/// graph declares no shape at all).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDecl {
    pub name: String,
    pub elem_type: i32,
    pub shape: Option<Vec<Dim>>,
}

/// The graph's inputs (those not backed by an initializer, which older
/// exporters also list as inputs) and its outputs, in declaration order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphInterface {
    pub inputs: Vec<TensorDecl>,
    pub outputs: Vec<TensorDecl>,
}

/// Read `path`'s interface.
pub fn read_interface(path: &Path) -> Result<GraphInterface, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("can't open {}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut r = Reader { inner: BufReader::new(file) };
    let not_onnx = |e: String| format!("{} isn't a readable ONNX model: {e}", path.display());
    let mut graph = None;
    r.fields(len, |r, field, wire, end| match (field, wire) {
        (7, LEN) => {
            graph = Some(r.graph(end)?);
            Ok(())
        }
        _ => r.skip(wire, end),
    })
    .map_err(not_onnx)?;
    let (mut interface, initializers) = graph.ok_or_else(|| not_onnx("no graph".into()))?;
    interface.inputs.retain(|i| !initializers.contains(&i.name));
    Ok(interface)
}

const VARINT: u8 = 0;
const I64: u8 = 1;
const LEN: u8 = 2;
const I32: u8 = 5;

struct Reader<R> {
    inner: R,
}

type Res<T> = Result<T, String>;

impl<R: Read + Seek> Reader<R> {
    fn pos(&mut self) -> Res<u64> {
        self.inner.stream_position().map_err(|e| e.to_string())
    }

    fn byte(&mut self) -> Res<u8> {
        let mut b = [0u8];
        self.inner.read_exact(&mut b).map_err(|_| "truncated".to_string())?;
        Ok(b[0])
    }

    fn varint(&mut self) -> Res<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("a varint longer than 10 bytes".into())
    }

    /// Call `f(field, wire type, end of the field's value)` for each field
    /// of the message that ends at `end`. `f` must consume the value.
    fn fields(&mut self, end: u64, mut f: impl FnMut(&mut Self, u64, u8, u64) -> Res<()>) -> Res<()> {
        while self.pos()? < end {
            let key = self.varint()?;
            let (field, wire) = (key >> 3, (key & 7) as u8);
            let value_end = match wire {
                LEN => {
                    let n = self.varint()?;
                    let end_of_value = self.pos()? + n;
                    if end_of_value > end {
                        return Err(format!("field {field} runs past its message"));
                    }
                    end_of_value
                }
                _ => end,
            };
            f(self, field, wire, value_end)?;
            if wire == LEN && self.pos()? != value_end {
                return Err(format!("field {field} was misread"));
            }
        }
        Ok(())
    }

    /// Step past a value we don't need.
    fn skip(&mut self, wire: u8, end: u64) -> Res<()> {
        match wire {
            VARINT => self.varint().map(|_| ()),
            I64 => self.seek_by(8),
            I32 => self.seek_by(4),
            LEN => self.inner.seek(SeekFrom::Start(end)).map(|_| ()).map_err(|e| e.to_string()),
            w => Err(format!("unsupported wire type {w}")),
        }
    }

    fn seek_by(&mut self, n: i64) -> Res<()> {
        self.inner.seek_relative(n).map_err(|e| e.to_string())
    }

    fn string(&mut self, end: u64) -> Res<String> {
        let n = (end - self.pos()?) as usize;
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf).map_err(|_| "truncated".to_string())?;
        String::from_utf8(buf).map_err(|_| "a name that isn't UTF-8".into())
    }

    /// GraphProto: inputs (11), outputs (12) and initializer names (5).
    fn graph(&mut self, end: u64) -> Res<(GraphInterface, HashSet<String>)> {
        let mut interface = GraphInterface::default();
        let mut initializers = HashSet::new();
        self.fields(end, |r, field, wire, end| match (field, wire) {
            (11, LEN) => {
                interface.inputs.push(r.value_info(end)?);
                Ok(())
            }
            (12, LEN) => {
                interface.outputs.push(r.value_info(end)?);
                Ok(())
            }
            (5, LEN) => {
                // TensorProto: its name (8); its data is skipped.
                let mut name = None;
                r.fields(end, |r, field, wire, end| match (field, wire) {
                    (8, LEN) => {
                        name = Some(r.string(end)?);
                        Ok(())
                    }
                    _ => r.skip(wire, end),
                })?;
                initializers.extend(name);
                Ok(())
            }
            _ => r.skip(wire, end),
        })?;
        Ok((interface, initializers))
    }

    /// ValueInfoProto: name (1), type (2).
    fn value_info(&mut self, end: u64) -> Res<TensorDecl> {
        let mut decl = TensorDecl { name: String::new(), elem_type: 0, shape: None };
        self.fields(end, |r, field, wire, end| match (field, wire) {
            (1, LEN) => {
                decl.name = r.string(end)?;
                Ok(())
            }
            // TypeProto: tensor_type (1); other kinds stay elem_type 0.
            (2, LEN) => r.fields(end, |r, field, wire, end| match (field, wire) {
                (1, LEN) => r.tensor_type(end, &mut decl),
                _ => r.skip(wire, end),
            }),
            _ => r.skip(wire, end),
        })?;
        Ok(decl)
    }

    /// TypeProto.Tensor: elem_type (1), shape (2).
    fn tensor_type(&mut self, end: u64, decl: &mut TensorDecl) -> Res<()> {
        self.fields(end, |r, field, wire, end| match (field, wire) {
            (1, VARINT) => {
                decl.elem_type = r.varint()? as i32;
                Ok(())
            }
            (2, LEN) => {
                let mut dims = Vec::new();
                // TensorShapeProto: dim (1), each a Dimension: dim_value (1)
                // or dim_param (2).
                r.fields(end, |r, field, wire, end| match (field, wire) {
                    (1, LEN) => {
                        let mut dim = Dim::Unknown;
                        r.fields(end, |r, field, wire, end| match (field, wire) {
                            (1, VARINT) => {
                                dim = Dim::Fixed(r.varint()? as i64);
                                Ok(())
                            }
                            (2, LEN) => {
                                dim = Dim::Symbolic(r.string(end)?);
                                Ok(())
                            }
                            _ => r.skip(wire, end),
                        })?;
                        dims.push(dim);
                        Ok(())
                    }
                    _ => r.skip(wire, end),
                })?;
                decl.shape = Some(dims);
                Ok(())
            }
            _ => r.skip(wire, end),
        })
    }
}

/// Protobuf encoding, for tests that build ONNX headers by hand.
#[cfg(test)]
pub(crate) mod encode {
    pub fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    pub fn len_field(field: u64, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(field << 3 | 2, &mut out);
        varint(body.len() as u64, &mut out);
        out.extend_from_slice(body);
        out
    }

    pub fn varint_field(field: u64, v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        varint(field << 3, &mut out);
        varint(v, &mut out);
        out
    }

    /// A ValueInfoProto: a tensor with `elem_type` and `dims` (a number is
    /// fixed, anything else a name).
    pub fn tensor(name: &str, elem_type: i32, dims: &[&str]) -> Vec<u8> {
        let shape: Vec<u8> = dims
            .iter()
            .flat_map(|d| {
                let dim = match d.parse::<u64>() {
                    Ok(n) => varint_field(1, n),
                    Err(_) => len_field(2, d.as_bytes()),
                };
                len_field(1, &dim)
            })
            .collect();
        let tensor_type = [varint_field(1, elem_type as u64), len_field(2, &shape)].concat();
        [len_field(1, name.as_bytes()), len_field(2, &len_field(1, &tensor_type))].concat()
    }

    /// A ModelProto with an IR version, a node, `inputs`, `outputs`, and an
    /// initializer named `weight` with `weight_bytes` of raw data.
    pub fn model(inputs: &[Vec<u8>], outputs: &[Vec<u8>], weight_bytes: usize) -> Vec<u8> {
        let initializer = [len_field(8, b"weight"), len_field(9, &vec![7u8; weight_bytes])].concat();
        let mut graph = len_field(1, &len_field(4, b"MatMul"));
        graph.extend(len_field(5, &initializer));
        for i in inputs {
            graph.extend(len_field(11, i));
        }
        for o in outputs {
            graph.extend(len_field(12, o));
        }
        [varint_field(1, 8), len_field(2, b"test"), len_field(7, &graph)].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::encode::*;
    use super::*;

    fn write(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("sk-onnx-graph-{tag}-{}.onnx", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn reads_inputs_and_outputs_and_skips_weights() {
        let model = model(
            &[
                tensor("input_ids", ONNX_INT64, &["batch", "sequence"]),
                tensor("attention_mask", ONNX_INT64, &["batch", "sequence"]),
                // An exporter that lists its initializers as inputs too.
                tensor("weight", ONNX_FLOAT, &["384", "384"]),
            ],
            &[tensor("last_hidden_state", ONNX_FLOAT, &["batch", "sequence", "384"])],
            1 << 20,
        );
        let path = write("ok", &model);
        let g = read_interface(&path).unwrap();
        let names = |v: &[TensorDecl]| v.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&g.inputs), ["input_ids", "attention_mask"], "the initializer isn't an input");
        assert_eq!(g.inputs[0].elem_type, ONNX_INT64);
        assert_eq!(g.inputs[0].shape, Some(vec![Dim::Symbolic("batch".into()), Dim::Symbolic("sequence".into())]));
        assert_eq!(names(&g.outputs), ["last_hidden_state"]);
        assert_eq!(g.outputs[0].shape.as_ref().unwrap()[2], Dim::Fixed(384));
        std::fs::remove_file(&path).unwrap();
    }

    /// A large model (`SIDEKICK_ONNX_BIG_FILE`, written beforehand, with its
    /// weights stored inline before its inputs, as exporters write
    /// GraphProto), read without reading the weights. Run by hand, under
    /// `/usr/bin/time -l`: `cargo test -p sidekick-core --release
    /// reads_a_large -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn reads_a_large_model_without_reading_its_weights() {
        let path = std::path::PathBuf::from(std::env::var("SIDEKICK_ONNX_BIG_FILE").expect("SIDEKICK_ONNX_BIG_FILE"));
        let size = std::fs::metadata(&path).unwrap().len();
        let started = std::time::Instant::now();
        let g = read_interface(&path).unwrap();
        println!("{} MB model: {} inputs, {} outputs, read in {:?}", size >> 20, g.inputs.len(), g.outputs.len(), started.elapsed());
        assert!(!g.inputs.is_empty() && !g.outputs.is_empty());
    }

    #[test]
    fn a_damaged_file_is_an_error_not_a_panic() {
        let model = model(&[tensor("input_ids", ONNX_INT64, &["1", "8"])], &[], 64);
        for (tag, bytes) in [("cut", &model[..model.len() - 10]), ("noise", &b"not a model at all"[..]), ("empty", &[][..])] {
            let path = write(tag, bytes);
            let e = read_interface(&path).unwrap_err();
            assert!(e.contains("isn't a readable ONNX model"), "{tag}: {e}");
            std::fs::remove_file(&path).unwrap();
        }
    }
}
