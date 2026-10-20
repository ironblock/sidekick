//! One copy of a model's tokenizer per process. A model's backend and the
//! daemon's `/tokenize` both need it; each loads it through a [`Shared`],
//! which hands out the copy already in use, or loads one, once: a second
//! caller for the same model waits for the first's load instead of
//! loading its own. Entries are weak, so a tokenizer nobody holds is
//! dropped with its last user, as before.

use sidekick_core::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// One model's entry: the copy in use, if any; locked while it loads.
type Slot<T> = Arc<Mutex<Weak<T>>>;

/// A process-wide cache of `T`, by model directory and id.
pub(crate) struct Shared<T> {
    slots: Mutex<HashMap<(PathBuf, String), Slot<T>>>,
}

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self { slots: Mutex::new(HashMap::new()) }
    }
}

impl<T> Shared<T> {
    /// The model's `T` in use, or `load`'s. Loads of one model are one at
    /// a time; other models' don't wait on them. Runs on a blocking thread.
    pub(crate) fn get_or_load(&self, dir: &Path, id: &str, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        let slot = {
            let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
            // Forget models nobody holds; a slot mid-load is locked, so kept.
            slots.retain(|_, slot| slot.try_lock().map_or(true, |weak| weak.strong_count() > 0));
            slots.entry((dir.to_path_buf(), id.to_string())).or_default().clone()
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
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn one_load_while_any_caller_holds_it() {
        let shared: Arc<Shared<String>> = Arc::default();
        let loads = Arc::new(AtomicUsize::new(0));
        let load = |loads: Arc<AtomicUsize>| {
            move || {
                std::thread::sleep(Duration::from_millis(50));
                loads.fetch_add(1, Ordering::SeqCst);
                Ok("tok".to_string())
            }
        };
        // Concurrent cold loads of one model: one load, one copy.
        let copies: Vec<_> = (0..4)
            .map(|_| {
                let (shared, loads) = (shared.clone(), loads.clone());
                std::thread::spawn(move || shared.get_or_load(Path::new("/m"), "a", load(loads)).unwrap())
            })
            .map(|t| t.join().unwrap())
            .collect();
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(copies.iter().all(|c| Arc::ptr_eq(c, &copies[0])));
        // Another model loads its own.
        shared.get_or_load(Path::new("/m"), "b", load(loads.clone())).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        // Once nobody holds it, it's gone, and the next caller loads again.
        drop(copies);
        shared.get_or_load(Path::new("/m"), "a", load(loads.clone())).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }
}
