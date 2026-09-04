// Denormal (subnormal) float handling, exposed via `std/dsp/denormal`.
//
// On most cores, a denormal operand takes a slow path in the FPU. A filter or a
// delay tail that decays to silence is a signal of almost only denormals, so it
// can take several times longer to process than a block of normal audio. To
// avoid this, real-time audio code tells the FPU to flush denormals to zero for
// the length of a block instead of to compute them.
//
//   x86: MXCSR bit 15 (FTZ, flush a denormal *result* to zero) and bit 6 (DAZ,
//        read a denormal *input* as zero). SSE2 is baseline on x86-64.
//   aarch64: FPCR bit 24 (FZ) does both.
//   other targets: no operation. The block runs at the denormal speed of that
//        target, which is still correct.
//
// The setting is per thread, and the thread is shared with the caller. A plugin
// runs on the audio thread of the host and must give it back unchanged. So
// `disable` returns the previous control word, and `restore` writes it back.
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__) \
    || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <xmmintrin.h>
#define RT_DENORMAL_MXCSR 1
#endif

uint64_t rt_denormals_disable(void) {
#if defined(RT_DENORMAL_MXCSR)
    unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040u); // (1 << 15) FTZ | (1 << 6) DAZ
    return (uint64_t)csr;
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr | (1ull << 24)));
    return fpcr;
#else
    return 0;
#endif
}

void rt_denormals_restore(uint64_t saved) {
#if defined(RT_DENORMAL_MXCSR)
    _mm_setcsr((unsigned int)saved);
#elif defined(__aarch64__)
    __asm__ __volatile__("msr fpcr, %0" : : "r"(saved));
#else
    (void)saved;
#endif
}