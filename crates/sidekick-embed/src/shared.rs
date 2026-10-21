//! One copy of a model's tokenizer per process. A model's backend and the
//! daemon's `/tokenize` both need it; each loads it through a [`Shared`],
//! which hands out the copy already in use, or loads one, once: a second
//! caller for the same model waits for the first's load instead of
//! loading its own. Entries are weak, so a tokenizer nobody holds is
//! dropped with its last user, as before.
//!
//! A copy is keyed by the model's directory and a fingerprint of its
//! manifest as resolved (prefixes, backend, `max_seq_len` after any cap),
//! so a model whose manifest reads differently gets a copy of its own. The
//! tokenizer.json itself isn't part of the key: a file replaced under a
//! model in use is read again only once nothing holds the old copy.

use sidekick_core::Result;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// One model's entry: the copy in use, if any; locked while it loads.
type Slot<T> = Arc<Mutex<Weak<T>>>;

/// A process-wide cache of `T`, by model directory and manifest.
pub(crate) struct Shared<T> {
    slots: Mutex<HashMap<(PathBuf, u64), Slot<T>>>,
}

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self { slots: Mutex::new(HashMap::new()) }
    }
}

/// A fingerprint of a resolved manifest, for the cache key.
pub(crate) fn fingerprint(manifest: &impl Debug) -> u64 {
    let mut hasher = DefaultHasher::new();
    format!("{manifest:?}").hash(&mut hasher);
    hasher.finish()
}

impl<T> Shared<T> {
    /// The model's `T` in use, or `load`'s. Loads of one model are one at
    /// a time; other models' don't wait on them. Runs on a blocking thread.
    pub(crate) fn get_or_load(&self, dir: &Path, manifest: u64, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        let slot = {
            let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
            // Forget models nobody holds. A slot another caller has cloned
            // (to load into, or to read) stays, or its load would land in a
            // slot the map no longer has, and the next caller load again.
            slots.retain(|_, slot| {
                Arc::strong_count(slot) > 1 || slot.try_lock().map_or(true, |weak| weak.strong_count() > 0)
            });
            slots.entry((dir.to_path_buf(), manifest)).or_default().clone()
        };
        let mut weak = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(value) = weak.upgrade() {
            return Ok(value);
        }
        let value = Arc::new(load()?);
        *weak = Arc::downgrade(&value);
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::Shared;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    fn counted(loads: &Arc<AtomicUsize>, wait: Duration) -> impl FnOnce() -> sidekick_core::Result<String> {
        let loads = loads.clone();
        move || {
            std::thread::sleep(wait);
            loads.fetch_add(1, Ordering::SeqCst);
            Ok("tok".to_string())
        }
    }

    #[test]
    fn one_load_while_any_caller_holds_it() {
        let shared: Arc<Shared<String>> = Arc::default();
        let loads = Arc::new(AtomicUsize::new(0));
        let wait = Duration::from_millis(50);
        // Concurrent cold loads of one model: one load, one copy.
        let copies: Vec<_> = (0..4)
            .map(|_| {
                let (shared, loads) = (shared.clone(), loads.clone());
                std::thread::spawn(move || shared.get_or_load(Path::new("/m"), 1, counted(&loads, wait)).unwrap())
            })
            .map(|t| t.join().unwrap())
            .collect();
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(copies.iter().all(|c| Arc::ptr_eq(c, &copies[0])));
        // Another model, or the same directory with another manifest, loads
        // its own.
        let _b = shared.get_or_load(Path::new("/m"), 2, counted(&loads, wait)).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        // Once nobody holds it, it's gone, and the next caller loads again.
        drop(copies);
        shared.get_or_load(Path::new("/m"), 1, counted(&loads, wait)).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_load_in_progress_survives_another_models_lookup() {
        // A cold load of `a` takes its slot; while it loads, a lookup of
        // `b` prunes. `a`'s slot must survive, so a later caller of `a`
        // gets the loaded copy rather than loading a second.
        let shared: Arc<Shared<String>> = Arc::default();
        let loads = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(Barrier::new(2));
        let a = {
            let (shared, loads, started) = (shared.clone(), loads.clone(), started.clone());
            std::thread::spawn(move || {
                let load = counted(&loads, Duration::from_millis(100));
                shared
                    .get_or_load(Path::new("/m"), 1, move || {
                        started.wait();
                        load()
                    })
                    .unwrap()
            })
        };
        started.wait();
        let _b = shared.get_or_load(Path::new("/m"), 2, counted(&loads, Duration::ZERO)).unwrap();
        let a = a.join().unwrap();
        let again = shared.get_or_load(Path::new("/m"), 1, counted(&loads, Duration::ZERO)).unwrap();
        assert!(Arc::ptr_eq(&a, &again));
        assert_eq!(loads.load(Ordering::SeqCst), 2, "a and b, once each");
    }
}
