//! How many threads ONNX Runtime runs a model on (D40): the machine's
//! fastest physical cores, capped at 8 unless the model raises the cap.

/// Inference threads per model when the machine's fastest cores can't be
/// detected: [`FALLBACK_INTRA_OP_THREADS`], or one less than its logical
/// CPUs if that's fewer, and at least 1.
pub const FALLBACK_INTRA_OP_THREADS: usize = 4;

/// The default's cap: past 8 threads, latency stopped improving on every
/// machine measured, except for the largest model, which may raise it
/// (`[onnx] default_intra_op_threads`; D40).
pub const MAX_DEFAULT_INTRA_OP_THREADS: Option<usize> = Some(8);

/// A model's default inference threads, when the operator configured none:
/// `min(max(cap, model_default), cores)`. The cap is
/// [`MAX_DEFAULT_INTRA_OP_THREADS`]. A model's own default can raise it,
/// never lower it, and nothing exceeds the machine's fast cores.
pub fn default_threads(cores: usize, model_default: Option<usize>) -> usize {
    let cap = MAX_DEFAULT_INTRA_OP_THREADS.unwrap_or(usize::MAX);
    cap.max(model_default.unwrap_or(0)).min(cores).max(1)
}

/// The machine's fastest tier of physical cores, and how they were found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectedCores {
    /// `hw.perflevel0.physicalcpu` (Apple silicon's P-cores),
    /// `hw.physicalcpu` (an Intel Mac), `linux cpu_core` (a hybrid Intel
    /// CPU's P-cores), `linux topology` (physical cores), or `fallback`.
    pub source: &'static str,
    pub cores: usize,
}

/// The default intra-op threads on this machine, for a model without its
/// own (D40): its fast physical cores, capped at 8. One thread per physical
/// core of the fastest tier means ONNX Runtime's threads don't share a core
/// (SMT) or wait on an efficiency core. Never the logical CPU count: on a
/// 12-core, 24-thread Xeon, 24 threads ran 1.4-2.9x slower than 8.
pub fn default_intra_op_threads() -> usize {
    default_threads(detected_cores().cores, None)
}

/// The machine's fastest tier of physical cores, detected once; if nothing
/// can be detected, the fallback rule's count.
pub fn detected_cores() -> DetectedCores {
    static DETECTED: std::sync::OnceLock<DetectedCores> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| {
        detect().unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
            DetectedCores { source: "fallback", cores: FALLBACK_INTRA_OP_THREADS.min(cpus.saturating_sub(1)).max(1) }
        })
    })
}

#[cfg(target_os = "macos")]
fn detect() -> Option<DetectedCores> {
    let sysctl = |name: &str| -> Option<usize> {
        let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", name]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).trim().parse().ok().filter(|&n: &usize| n > 0)
    };
    // Apple silicon reports its performance cores as perflevel0; an Intel
    // Mac has no perflevels.
    match sysctl("hw.perflevel0.physicalcpu") {
        Some(cores) => Some(DetectedCores { source: "hw.perflevel0.physicalcpu", cores }),
        None => sysctl("hw.physicalcpu").map(|cores| DetectedCores { source: "hw.physicalcpu", cores }),
    }
}

#[cfg(target_os = "linux")]
fn detect() -> Option<DetectedCores> {
    linux::detect(std::path::Path::new("/"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect() -> Option<DetectedCores> {
    None
}

/// Physical cores from sysfs, under `root` (`/` on a running system), so
/// tests can give it fixture trees.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod linux {
    use super::DetectedCores;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// A hybrid Intel CPU's P-cores (`/sys/devices/cpu_core/cpus`), else
    /// every physical core; each physical core counted once, however many
    /// SMT threads it has.
    pub fn detect(root: &Path) -> Option<DetectedCores> {
        let cpu = root.join("sys/devices/system/cpu");
        if let Ok(list) = std::fs::read_to_string(root.join("sys/devices/cpu_core/cpus")) {
            if let Some(cores) = parse_cpu_list(list.trim()).and_then(|cpus| physical_cores(&cpu, &cpus)) {
                return Some(DetectedCores { source: "linux cpu_core", cores });
            }
        }
        let cpus: Vec<usize> = std::fs::read_dir(&cpu)
            .ok()?
            .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("cpu")?.parse().ok())
            .collect();
        physical_cores(&cpu, &cpus).map(|cores| DetectedCores { source: "linux topology", cores })
    }

    /// Unique (package, core) pairs among `cpus`; `None` if any CPU's
    /// topology can't be read.
    fn physical_cores(cpu: &Path, cpus: &[usize]) -> Option<usize> {
        let read = |n: usize, file: &str| -> Option<i64> {
            std::fs::read_to_string(cpu.join(format!("cpu{n}/topology/{file}"))).ok()?.trim().parse().ok()
        };
        let mut cores = BTreeSet::new();
        for &n in cpus {
            cores.insert((read(n, "physical_package_id")?, read(n, "core_id")?));
        }
        (!cores.is_empty()).then_some(cores.len())
    }

    /// A kernel CPU list: `0-7,16-23`.
    pub fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for part in list.split(',').filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => cpus.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
                None => cpus.push(part.parse().ok()?),
            }
        }
        Some(cpus)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A sysfs tree: each (cpu, package, core) gets its topology files.
        fn tree(tag: &str, cpus: &[(usize, i64, i64)], cpu_core: Option<&str>) -> std::path::PathBuf {
            let root = std::env::temp_dir().join(format!("sk-sysfs-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for &(n, package, core) in cpus {
                let t = root.join(format!("sys/devices/system/cpu/cpu{n}/topology"));
                std::fs::create_dir_all(&t).unwrap();
                std::fs::write(t.join("physical_package_id"), format!("{package}\n")).unwrap();
                std::fs::write(t.join("core_id"), format!("{core}\n")).unwrap();
            }
            // Entries that aren't CPUs.
            std::fs::create_dir_all(root.join("sys/devices/system/cpu/cpufreq")).unwrap();
            if let Some(list) = cpu_core {
                std::fs::create_dir_all(root.join("sys/devices/cpu_core")).unwrap();
                std::fs::write(root.join("sys/devices/cpu_core/cpus"), format!("{list}\n")).unwrap();
            }
            root
        }

        #[test]
        fn smt_siblings_count_once() {
            // 12 cores, 2 threads each: cpu n and n+12 share core n.
            let cpus: Vec<_> = (0..24).map(|n| (n, 0, (n % 12) as i64)).collect();
            let root = tree("smt", &cpus, None);
            assert_eq!(detect(&root), Some(DetectedCores { source: "linux topology", cores: 12 }));
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn two_packages_count_their_cores_apart() {
            let cpus: Vec<_> = (0..8).map(|n| (n, (n / 4) as i64, (n % 4) as i64)).collect();
            let root = tree("packages", &cpus, None);
            assert_eq!(detect(&root).unwrap().cores, 8, "core ids repeat across packages");
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn a_hybrid_cpu_counts_its_p_cores() {
            // 6 P-cores with SMT (cpus 0-11, two per core) and 8 E-cores
            // (cpus 12-19).
            let mut cpus: Vec<_> = (0..12).map(|n| (n, 0, (n / 2) as i64)).collect();
            cpus.extend((12..20).map(|n| (n, 0, 100 + n as i64)));
            let root = tree("hybrid", &cpus, Some("0-11"));
            assert_eq!(detect(&root), Some(DetectedCores { source: "linux cpu_core", cores: 6 }));
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn missing_files_detect_nothing() {
            let root = tree("missing", &[], None);
            assert_eq!(detect(&root), None);
            // A CPU without its topology files.
            std::fs::create_dir_all(root.join("sys/devices/system/cpu/cpu0")).unwrap();
            assert_eq!(detect(&root), None);
            // An unreadable cpu_core list falls back to the topology.
            let root2 = tree("bad-list", &[(0, 0, 0), (1, 0, 1)], Some("x-y"));
            assert_eq!(detect(&root2), Some(DetectedCores { source: "linux topology", cores: 2 }));
            for r in [root, root2] {
                std::fs::remove_dir_all(&r).unwrap();
            }
        }

        #[test]
        fn the_default_is_capped_at_8_unless_the_model_raises_it_within_the_cores() {
            use super::super::default_threads;
            assert_eq!(default_threads(4, Some(16)), 4, "never past the fast cores");
            assert_eq!(default_threads(32, None), 8, "the cap");
            assert_eq!(default_threads(32, Some(16)), 16, "a model raises it");
            assert_eq!(default_threads(12, Some(4)), 8, "never below the cap");
            assert_eq!(default_threads(8, None), 8);
        }

        #[test]
        fn cpu_lists_parse() {
            assert_eq!(parse_cpu_list("0-3,8,10-11"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
            assert_eq!(parse_cpu_list(""), Some(vec![]));
            assert_eq!(parse_cpu_list("a"), None);
        }
    }
}
