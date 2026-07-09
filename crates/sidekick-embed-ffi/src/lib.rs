//! C ABI for sidekick's embedding tiers: `libsidekick.dylib`.
//!
//! For host applications that want on-device embeddings (Core ML/ANE
//! encoders, static floor) *in-process* — no daemon, no HTTP. The daemon
//! remains the primary interface; this exists for the "sidekick installed
//! but sidekickd not running" integration path (see docs/INTEGRATING.md and
//! DECISIONS.md D18). Chat is deliberately not exposed: it would drag the
//! FoundationModels Swift shim into every host, and a conversational
//! session wants a daemon lifetime anyway.
//!
//! Contract (also documented in include/sidekick.h):
//! - Every entry point is panic-safe: panics are caught and reported as
//!   errors, never unwound across the FFI boundary.
//! - Strings in and out are UTF-8. Out-strings are freed with
//!   `sk_string_free`, embedding buffers with `sk_floats_free`.
//! - `sk_pool` is thread-safe. The first `sk_embed` for a model loads it
//!   (Core ML: up to seconds for large encoders) with the model map
//!   unlocked, so calls on other, already-loaded models are unaffected;
//!   loaded models stay resident until `sk_pool_close`.

use sidekick_core::{truncate_normalized, EmbedPurpose, Embedder, ModelRegistry};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

/// Bumped on any breaking change to this ABI.
pub const SK_ABI_VERSION: u32 = 1;

pub struct SkPool {
    registry: ModelRegistry,
    loaded: Mutex<HashMap<String, Arc<dyn Embedder>>>,
}

fn set_err(err: *mut *mut c_char, message: impl std::fmt::Display) {
    if err.is_null() {
        return;
    }
    let s = message.to_string();
    // NUL bytes inside the message would truncate it; strip them.
    let c = CString::new(s.replace('\0', " ")).unwrap_or_default();
    unsafe { *err = c.into_raw() };
}

/// Run `f` with panics converted to an error string. `default` is returned
/// on panic (a null/zero of the appropriate type).
fn ffi_guard<T>(err: *mut *mut c_char, default: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_err(err, e);
            default
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic in sidekick_embed".into());
            set_err(err, format!("internal panic: {msg}"));
            default
        }
    }
}

unsafe fn utf8<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err(format!("{what} is NULL"));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| format!("{what} is not valid UTF-8"))
}

fn default_models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("sidekick")
        .join("models")
}

/// ABI version of this library. Check it before anything else.
#[no_mangle]
pub extern "C" fn sk_abi_version() -> u32 {
    SK_ABI_VERSION
}

/// Open a pool over a models directory (NULL → the daemon's default,
/// `~/Library/Application Support/sidekick/models` on macOS). Returns NULL
/// on failure with `*err` set (free with `sk_string_free`). An empty or
/// missing directory is not an error — `sk_pool_models` just returns `[]`.
///
/// # Safety
/// `models_dir` must be NULL or a valid NUL-terminated string; `err` must be
/// NULL or a valid out-pointer.
#[no_mangle]
pub unsafe extern "C" fn sk_pool_open(
    models_dir: *const c_char,
    err: *mut *mut c_char,
) -> *mut SkPool {
    ffi_guard(err, std::ptr::null_mut(), || {
        let dir = if models_dir.is_null() {
            default_models_dir()
        } else {
            PathBuf::from(utf8(models_dir, "models_dir")?)
        };
        let registry = ModelRegistry::scan(&dir).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(SkPool {
            registry,
            loaded: Mutex::new(HashMap::new()),
        })))
    })
}

/// Close a pool, dropping every resident model. `pool` may be NULL.
///
/// # Safety
/// `pool` must be NULL or a pointer from `sk_pool_open`, not used afterward.
#[no_mangle]
pub unsafe extern "C" fn sk_pool_close(pool: *mut SkPool) {
    if !pool.is_null() {
        // A panicking Drop would otherwise unwind out of extern "C" = abort.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(pool))));
    }
}

/// JSON array of available model ids, e.g.
/// `["bge-small-en-v1.5","static-floor"]`. Free with `sk_string_free`.
/// Returns NULL on failure with `*err` set. An empty models directory is
/// `"[]"`, not NULL.
///
/// # Safety
/// Pointer rules as above.
#[no_mangle]
pub unsafe extern "C" fn sk_pool_models(
    pool: *const SkPool,
    err: *mut *mut c_char,
) -> *mut c_char {
    ffi_guard(err, std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is NULL")?;
        let ids: Vec<&str> = pool.registry.ids().collect();
        let json = serde_json::to_string(&ids).map_err(|e| e.to_string())?;
        Ok(CString::new(json).unwrap_or_default().into_raw())
    })
}

/// JSON description of one model from its manifest (the model is not
/// loaded): `{"id","backend","dims","matryoshka","max_seq_len"}`.
/// `matryoshka` lists the dims values `sk_embed` accepts as
/// `requested_dims`; empty means native dims only. Free with
/// `sk_string_free`. Returns NULL on failure with `*err` set.
///
/// # Safety
/// Pointer rules as above.
#[no_mangle]
pub unsafe extern "C" fn sk_model_info(
    pool: *const SkPool,
    model_id: *const c_char,
    err: *mut *mut c_char,
) -> *mut c_char {
    ffi_guard(err, std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is NULL")?;
        let id = utf8(model_id, "model_id")?;
        let m = &pool.registry.get(id).map_err(|e| e.to_string())?.manifest;
        let json = serde_json::json!({
            "id": m.id,
            "backend": m.backend,
            "dims": m.dims,
            "matryoshka": m.matryoshka,
            "max_seq_len": m.max_seq_len,
        });
        Ok(CString::new(json.to_string()).unwrap_or_default().into_raw())
    })
}

/// Native output dimensionality of a model (from its manifest — the model is
/// not loaded). Returns 0 for an unknown id, with `*err` set.
///
/// # Safety
/// Pointer rules as above.
#[no_mangle]
pub unsafe extern "C" fn sk_embed_dims(
    pool: *const SkPool,
    model_id: *const c_char,
    err: *mut *mut c_char,
) -> usize {
    ffi_guard(err, 0, || {
        let pool = pool.as_ref().ok_or("pool is NULL")?;
        let id = utf8(model_id, "model_id")?;
        let model = pool.registry.get(id).map_err(|e| e.to_string())?;
        Ok(model.manifest.dims)
    })
}

/// Embed `n_texts` UTF-8 strings with a model. `purpose`: 0 = document,
/// 1 = query (applies the model's query prefix). `requested_dims`: 0 for
/// the model's native dims, or one of its matryoshka dims (see
/// `sk_model_info`) to get truncated + renormalized vectors — the same
/// semantics as the daemon's `dimensions` parameter. On success returns a
/// buffer of `n_texts * *dims_out` floats (row-major, unit-normalized rows)
/// that the caller frees with `sk_floats_free(ptr, n_texts * dims)`.
/// Returns NULL on failure with `*err` set.
///
/// The first call for a model loads it (Core ML: seconds for large
/// encoders) with the model map unlocked — concurrent calls on other
/// models proceed; two concurrent first-calls on the same model may both
/// load, and the loser's copy is dropped. Models stay resident until
/// `sk_pool_close`.
///
/// # Safety
/// `texts` must point to `n_texts` valid NUL-terminated strings; `dims_out`
/// must be a valid out-pointer; other pointer rules as above.
#[no_mangle]
pub unsafe extern "C" fn sk_embed(
    pool: *const SkPool,
    model_id: *const c_char,
    texts: *const *const c_char,
    n_texts: usize,
    purpose: c_int,
    requested_dims: usize,
    dims_out: *mut usize,
    err: *mut *mut c_char,
) -> *mut f32 {
    ffi_guard(err, std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is NULL")?;
        let id = utf8(model_id, "model_id")?;
        if dims_out.is_null() {
            return Err("dims_out is NULL".into());
        }
        if texts.is_null() && n_texts > 0 {
            return Err("texts is NULL".into());
        }
        let purpose = match purpose {
            0 => EmbedPurpose::Document,
            1 => EmbedPurpose::Query,
            other => return Err(format!("purpose must be 0 (document) or 1 (query), got {other}")),
        };
        let mut owned = Vec::with_capacity(n_texts);
        for i in 0..n_texts {
            owned.push(utf8(*texts.add(i), "text")?);
        }

        // Look up under the lock; load with it RELEASED so a multi-second
        // Core ML load can't stall calls on other, already-loaded models
        // (same pattern as the daemon's EmbedderPool). A poisoned lock is
        // recovered, not propagated: the critical sections only read or
        // insert-as-last-op, so no invariant can be left broken.
        let cached = pool
            .loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned();
        let embedder = match cached {
            Some(e) => e,
            None => {
                let model = pool.registry.get(id).map_err(|e| e.to_string())?;
                let loaded_model: Arc<dyn Embedder> =
                    Arc::from(sidekick_embed::load_embedder(model).map_err(|e| e.to_string())?);
                pool.loaded
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(id.to_string())
                    .or_insert(loaded_model)
                    .clone()
            }
        };

        let native_dims = embedder.dims();
        let out_dims = match requested_dims {
            0 => native_dims,
            d if d == native_dims => native_dims,
            d if embedder.matryoshka_dims().contains(&d) => d,
            d => {
                return Err(format!(
                    "model `{id}` supports dimensions {:?}, got {d}",
                    if embedder.matryoshka_dims().is_empty() {
                        vec![native_dims]
                    } else {
                        embedder.matryoshka_dims().to_vec()
                    }
                ))
            }
        };

        let vectors = embedder.embed(&owned, purpose).map_err(|e| e.to_string())?;
        if vectors.len() != n_texts {
            // Never violate the n_texts * dims buffer contract the caller
            // frees/reads with — a mismatched length would be heap UB.
            return Err(format!(
                "embedder returned {} vectors for {n_texts} texts",
                vectors.len()
            ));
        }
        let mut flat = Vec::with_capacity(n_texts * out_dims);
        for v in &vectors {
            if v.len() != native_dims {
                return Err(format!(
                    "embedder returned {} dims, expected {native_dims}",
                    v.len()
                ));
            }
            if out_dims == native_dims {
                flat.extend_from_slice(v);
            } else {
                flat.extend_from_slice(&truncate_normalized(v, out_dims));
            }
        }
        *dims_out = out_dims;
        // Boxed slice: len == capacity, so sk_floats_free can rebuild it
        // from (ptr, n) alone.
        Ok(Box::into_raw(flat.into_boxed_slice()) as *mut f32)
    })
}

/// Free a buffer returned by `sk_embed`. `n_floats` must be exactly
/// `n_texts * dims` from that call. `ptr` may be NULL.
///
/// # Safety
/// Must be called at most once per buffer, with the exact length.
#[no_mangle]
pub unsafe extern "C" fn sk_floats_free(ptr: *mut f32, n_floats: usize) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n_floats)))
        }));
    }
}

/// Free a string returned by this library. `ptr` may be NULL.
///
/// # Safety
/// Must be called at most once per string.
#[no_mangle]
pub unsafe extern "C" fn sk_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(CString::from_raw(ptr))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Write a tiny static-floor model + manifest, mirroring the server's
    /// integration-test fixture.
    fn fixture_dir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "sk-ffi-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let model_dir = dir.join("floor");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("manifest.toml"),
            r#"
id = "static-floor"
backend = "static"
artifact = "model.safetensors"
tokenizer = "tokenizer.json"
dims = 3
matryoshka = [3, 2]
max_seq_len = 16
"#,
        )
        .unwrap();
        // 4-token vocab, 3 dims. Tokenizer: whitespace WordLevel.
        let table: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0,
        ];
        let bytes: Vec<u8> = table.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 3], &bytes)
                .unwrap();
        let data = safetensors::serialize([("embeddings", view)], &None).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), data).unwrap();
        std::fs::write(
            model_dir.join("tokenizer.json"),
            r#"{
  "version": "1.0",
  "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}, "unk_token": "alpha"}
}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn end_to_end_over_the_c_abi() {
        assert_eq!(sk_abi_version(), SK_ABI_VERSION);

        let dir = fixture_dir();
        let dir_c = CString::new(dir.to_str().unwrap()).unwrap();
        let mut err: *mut c_char = std::ptr::null_mut();

        unsafe {
            let pool = sk_pool_open(dir_c.as_ptr(), &mut err);
            assert!(!pool.is_null(), "open failed");

            let models = sk_pool_models(pool, &mut err);
            let json = CStr::from_ptr(models).to_str().unwrap().to_string();
            sk_string_free(models);
            assert_eq!(json, r#"["static-floor"]"#);

            let id = CString::new("static-floor").unwrap();
            assert_eq!(sk_embed_dims(pool, id.as_ptr(), &mut err), 3);

            let info = sk_model_info(pool, id.as_ptr(), &mut err);
            let info_json: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(info).to_str().unwrap()).unwrap();
            sk_string_free(info);
            assert_eq!(info_json["dims"], 3);
            assert_eq!(info_json["matryoshka"], serde_json::json!([3, 2]));
            assert_eq!(info_json["backend"], "static");

            let t1 = CString::new("beta").unwrap();
            let t2 = CString::new("beta gamma").unwrap();
            let texts = [t1.as_ptr(), t2.as_ptr()];
            let mut dims = 0usize;
            let out = sk_embed(pool, id.as_ptr(), texts.as_ptr(), 2, 0, 0, &mut dims, &mut err);
            assert!(!out.is_null(), "embed failed");
            assert_eq!(dims, 3);
            let flat = std::slice::from_raw_parts(out, 2 * dims);
            // "beta" -> row 1 normalized = (0,1,0)
            assert_eq!(&flat[..3], &[0.0, 1.0, 0.0]);
            // "beta gamma" -> mean of rows 1,2 = (0,.5,.5), normalized
            let inv = 1.0 / 2.0f32.sqrt();
            assert!((flat[3] - 0.0).abs() < 1e-6);
            assert!((flat[4] - inv).abs() < 1e-6 && (flat[5] - inv).abs() < 1e-6);
            sk_floats_free(out, 2 * dims);

            // matryoshka truncation: requested_dims=2 renormalizes rows
            let mut dims2 = 0usize;
            let out2 = sk_embed(pool, id.as_ptr(), texts.as_ptr(), 2, 0, 2, &mut dims2, &mut err);
            assert!(!out2.is_null(), "truncated embed failed");
            assert_eq!(dims2, 2);
            let flat2 = std::slice::from_raw_parts(out2, 2 * dims2);
            // "beta" -> (0,1,0) truncated to (0,1), already unit
            assert!((flat2[0] - 0.0).abs() < 1e-6 && (flat2[1] - 1.0).abs() < 1e-6);
            // "beta gamma" -> (0, inv, inv) truncated to (0, inv) -> renorm (0, 1)
            assert!((flat2[2] - 0.0).abs() < 1e-6 && (flat2[3] - 1.0).abs() < 1e-6);
            sk_floats_free(out2, 2 * dims2);

            // undeclared dims errors
            let mut err_d: *mut c_char = std::ptr::null_mut();
            let mut dims3 = 0usize;
            let bad_dims =
                sk_embed(pool, id.as_ptr(), texts.as_ptr(), 2, 0, 5, &mut dims3, &mut err_d);
            assert!(bad_dims.is_null());
            assert!(!err_d.is_null());
            sk_string_free(err_d);

            // unknown model -> 0 dims + err string
            let bad = CString::new("nope").unwrap();
            let mut err2: *mut c_char = std::ptr::null_mut();
            assert_eq!(sk_embed_dims(pool, bad.as_ptr(), &mut err2), 0);
            assert!(!err2.is_null());
            sk_string_free(err2);

            sk_pool_close(pool);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn null_and_bad_inputs_error_instead_of_crashing() {
        let mut err: *mut c_char = std::ptr::null_mut();
        unsafe {
            // NULL pool
            let mut err0: *mut c_char = std::ptr::null_mut();
            assert!(sk_pool_models(std::ptr::null(), &mut err0).is_null());
            assert!(!err0.is_null());
            sk_string_free(err0);
            let id = CString::new("x").unwrap();
            assert_eq!(sk_embed_dims(std::ptr::null(), id.as_ptr(), &mut err), 0);
            assert!(!err.is_null());
            sk_string_free(err);

            // bad purpose
            let dir = fixture_dir();
            let dir_c = CString::new(dir.to_str().unwrap()).unwrap();
            let pool = sk_pool_open(dir_c.as_ptr(), std::ptr::null_mut());
            let model = CString::new("static-floor").unwrap();
            let t = CString::new("alpha").unwrap();
            let texts = [t.as_ptr()];
            let mut dims = 0usize;
            let mut err3: *mut c_char = std::ptr::null_mut();
            let out =
                sk_embed(pool, model.as_ptr(), texts.as_ptr(), 1, 7, 0, &mut dims, &mut err3);
            assert!(out.is_null());
            assert!(!err3.is_null());
            sk_string_free(err3);
            sk_pool_close(pool);
            let _ = std::fs::remove_dir_all(dir);

            // NULL pool close is a no-op
            sk_pool_close(std::ptr::null_mut());
            sk_floats_free(std::ptr::null_mut(), 0);
            sk_string_free(std::ptr::null_mut());
        }
    }
}
