//! AVX-512 for ONNX Runtime on an Intel Mac (D42).
//!
//! ONNX Runtime picks its CPU kernels once, on the first thread that needs
//! them, and takes the AVX-512 ones only if the XCR0 register says the OS
//! saves AVX-512 state. macOS turns that state on lazily, per thread, the
//! first time a thread runs an AVX-512 instruction. Until then XCR0 reads
//! as AVX-only, even on a CPU that has AVX-512 (`hw.optional.avx512f`), so
//! ONNX Runtime falls back to its AVX2/FMA kernels.
//!
//! [`enable_on_this_thread`] runs one AVX-512 instruction on the calling
//! thread when the OS reports AVX-512, so that the thread that loads the
//! runtime and its sessions reads XCR0 with the AVX-512 bits set. Linux
//! enables AVX-512 state eagerly and needs none of this.
//!
//! `SIDEKICK_ONNX_AVX512=0` turns it off, to compare the two kernel sets.

/// Whether this process enables AVX-512 for ONNX Runtime: the OS reports
/// AVX-512F and `SIDEKICK_ONNX_AVX512` isn't `0`. Always false off Intel
/// macOS.
pub fn enabled() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("SIDEKICK_ONNX_AVX512").is_none_or(|v| v != "0") && os_reports_avx512f()
        })
    }
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    {
        false
    }
}

/// On Intel macOS with AVX-512 [`enabled`], make the OS turn on AVX-512
/// state for the calling thread. Cheap after the first call on a thread.
pub fn enable_on_this_thread() {
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    if enabled() {
        // SAFETY: the OS reports AVX-512F, so the instruction exists. It
        // zeroes zmm0, which the clobber covers: code built without AVX-512
        // keeps nothing in zmm0's upper bits. vzeroupper then clears the
        // upper state so later SSE code pays no transition cost.
        unsafe {
            std::arch::asm!("vpxord zmm0, zmm0, zmm0", "vzeroupper", out("xmm0") _, options(nomem, nostack));
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn os_reports_avx512f() -> bool {
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> std::ffi::c_int;
    }
    let mut value: std::ffi::c_int = 0;
    let mut len = std::mem::size_of_val(&value);
    // SAFETY: a NUL-terminated name and an int-sized buffer, as this sysctl
    // returns; it's absent (an error) on Macs without AVX-512.
    let rc = unsafe {
        sysctlbyname(c"hw.optional.avx512f".as_ptr(), (&raw mut value).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    rc == 0 && value == 1
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    /// XCR0's AVX-512 state bits (opmask, ZMM0-15 upper halves, ZMM16-31),
    /// read as ONNX Runtime reads them.
    fn xcr0_avx512() -> bool {
        let (lo, _hi): (u32, u32);
        // SAFETY: xgetbv with ecx = 0 exists on every Intel Mac (OSXSAVE).
        unsafe { std::arch::asm!("xgetbv", in("ecx") 0, out("eax") lo, out("edx") _hi, options(nomem, nostack)) };
        lo & 0xE0 == 0xE0
    }

    #[test]
    fn enabling_sets_the_state_onnx_runtime_checks() {
        // A fresh thread, so nothing else in the test process has used
        // AVX-512 on it.
        std::thread::spawn(|| {
            super::enable_on_this_thread();
            assert_eq!(xcr0_avx512(), super::enabled(), "XCR0 should report AVX-512 exactly when it's enabled");
        })
        .join()
        .unwrap();
    }
}
