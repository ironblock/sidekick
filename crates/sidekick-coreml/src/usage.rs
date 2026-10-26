//! This process's memory as the kernel accounts it: its physical footprint
//! and its Neural Engine footprint, the ANE programs it holds.
//!
//! #187 measured the ANE's budget as shared by the whole machine: past
//! about 2.8 GiB of programs in active use, they reload on nearly every
//! call. A process can read its own share, which `/health` reports
//! (docs/design/open-llm.md, "Memory budget").

#![allow(unsafe_code)]

use std::ffi::{c_int, c_void};

/// `RUSAGE_INFO_V6`, which added the Neural Engine footprint.
const RUSAGE_INFO_V6: c_int = 6;
/// `RUSAGE_INFO_V4`: the physical footprint, on every macOS sidekick runs on.
const RUSAGE_INFO_V4: c_int = 4;

/// `struct rusage_info_v6` from `<sys/resource.h>`: 58 64-bit words. Only
/// the two fields read here are named; the rest keep the layout. V4's
/// layout is a prefix of V6's.
#[repr(C)]
struct RusageInfoV6 {
    uuid: [u8; 16],
    before_phys_footprint: [u64; 7],
    phys_footprint: u64,
    before_neural_footprint: [u64; 36],
    neural_footprint: u64,
    rest: [u64; 11],
}

const _: () = assert!(std::mem::size_of::<RusageInfoV6>() == 464);
const _: () = assert!(std::mem::offset_of!(RusageInfoV6, phys_footprint) == 72);
const _: () = assert!(std::mem::offset_of!(RusageInfoV6, neural_footprint) == 368);

// `proc_pid_rusage` exists since macOS 10.9, so a strong link is safe on
// every macOS sidekick supports (the Intel build's floor is 13.4). What's
// newer is the V6 flavor, and an older kernel refuses it with an error
// rather than a missing symbol.
extern "C" {
    fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
}

/// This process's memory, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    /// The physical footprint (`phys_footprint`): what Activity Monitor's
    /// Memory column shows, and what a leak of Core ML's outputs grows.
    pub phys_footprint: u64,
    /// The ANE programs this process holds (`ri_neural_footprint`): their
    /// weights, less any table gathered on the CPU. Not part of
    /// `phys_footprint`. `None` where the kernel doesn't report it.
    pub neural_footprint: Option<u64>,
}

/// Read this process's memory; `None` if the kernel refuses.
pub fn process_memory() -> Option<ProcessMemory> {
    let mut info = RusageInfoV6 {
        uuid: [0; 16],
        before_phys_footprint: [0; 7],
        phys_footprint: 0,
        before_neural_footprint: [0; 36],
        neural_footprint: 0,
        rest: [0; 11],
    };
    let pid = std::process::id() as c_int;
    let buffer = (&mut info as *mut RusageInfoV6).cast::<c_void>();
    // SAFETY: the buffer is a V6 record, the largest flavor asked for; the
    // kernel writes at most that flavor's size.
    if unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V6, buffer) } == 0 {
        return Some(ProcessMemory {
            phys_footprint: info.phys_footprint,
            neural_footprint: Some(info.neural_footprint),
        });
    }
    // SAFETY: as above; V4 is a prefix of V6.
    if unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V4, buffer) } == 0 {
        return Some(ProcessMemory {
            phys_footprint: info.phys_footprint,
            neural_footprint: None,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_process_has_a_footprint() {
        let m = super::process_memory().expect("proc_pid_rusage");
        assert!(m.phys_footprint > 0);
        // A process that has loaded nothing onto the ANE holds nothing there.
        assert!(matches!(m.neural_footprint, None | Some(0)), "{m:?}");
        // Allocating and touching memory grows the footprint.
        let v = vec![1u8; 64 << 20];
        let grown = super::process_memory().unwrap().phys_footprint;
        assert!(
            grown >= m.phys_footprint + (32 << 20),
            "{} -> {grown}",
            m.phys_footprint
        );
        drop(v);
    }
}
