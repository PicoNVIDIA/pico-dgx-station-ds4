// k7_e2_ptx.cuh -- PTX wrappers for the K7 E2 probe (packed scale columns: TMA load, tcgen05.cp, tcgen05.ld).
//
// Rule of this file: no instruction string is invented. Each one is copied from a header that this project's
// engines already compile for sm_103a (Garner's kernels and vendored DeepGEMM, in vllm/vllm-openai:v0.31.0), from
// upstream SGLang's sm_100 kernels, or from NVIDIA's PTX ISA manual, and the source is cited on the wrapper.
// Abbreviations used in the citations:
//   W8   <built tree>/kernels/w8-persist/include/w8/ptx.cuh and umma.cuh   (Garner, production on the box)
//   W3   <built tree>/kernels/w3-preattn/include/w3/ptx.cuh and mxf8_core.cuh
//   W1   <built tree>/kernels/w1-attn/include/w1/ptx.cuh
//   DG   <gracetier checkout>/reference/vllm-0.31.0/deep_gemm/include/deep_gemm/...   (vendored DeepGEMM)
//   SGL  sglang/python/sglang/kernels/jit/include/sgl_kernel/ptx/tcgen05.cuh           (upstream SGLang)
//   PTX  NVIDIA "PTX ISA 9.5" manual, docs.nvidia.com/cuda/parallel-thread-execution, read 2026-11-18
// The two decompressing copies (forms 1 to 4 below) exist ONLY in the manual: no header on the Mac uses them.
// Their grammar is PTX 9.7.18.9.2: tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc
// with .dst_fmt = {.b8x16} and .src_fmt = {.b6x16_p32, .b4x16_p64}.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

// The inline assembly goes through one macro so that a host compiler can syntax-check the C++ around it
// (the Mac check defines K7_ASM away). For nvcc it is exactly `asm volatile(...)`.
#ifndef K7_ASM
#define K7_ASM(...) asm volatile(__VA_ARGS__)
#endif

// Which tensor-memory copy is under test (one .so per form, so that an assembler rejection of one form
// cannot take the control down with it).
#ifndef K7_CP_FORM
#define K7_CP_FORM 0
#endif
#if K7_CP_FORM == 0    // the copy the scale path uses today: W8 umma.cuh:71, W3 ptx.cuh:163, SGL :348
#define K7_CP_STR "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
#define K7_CP_ROWS 32
#elif K7_CP_FORM == 1  // PTX 9.7.18.9.2 grammar; 4-bit to 8-bit decompress (PTX 9.7.18.9.1.1)
#define K7_CP_STR "tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [%0], %1;"
#define K7_CP_ROWS 32
#elif K7_CP_FORM == 2  // PTX 9.7.18.9.2 grammar; 6-bit to 8-bit decompress (PTX 9.7.18.9.1.2)
#define K7_CP_STR "tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [%0], %1;"
#define K7_CP_ROWS 32
#elif K7_CP_FORM == 3  // fallback shape if the assembler refuses form 1 (PTX example: .128x128b.b8x16.b6x16_p32)
#define K7_CP_STR "tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;"
#define K7_CP_ROWS 128
#elif K7_CP_FORM == 4  // fallback shape if the assembler refuses form 2
#define K7_CP_STR "tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [%0], %1;"
#define K7_CP_ROWS 128
#elif K7_CP_FORM == 5  // plain 128-row copy, the control for forms 3 and 4: SGL :341
#define K7_CP_STR "tcgen05.cp.cta_group::1.128x128b [%0], %1;"
#define K7_CP_ROWS 128
#else
#error "K7_CP_FORM must be 0..5"
#endif

namespace k7 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {  // W8 ptx.cuh:8
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ------------------------------------------------------------------ mbarrier (CTA-local)
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {  // W8 ptx.cuh:88-90
  K7_ASM("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {  // W8 ptx.cuh:35
  K7_ASM("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t bytes) {  // W8 ptx.cuh:97-99
  K7_ASM("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
// One try: true if the phase completed. The last operand is the suspend-time hint in ns (10 ms).
__device__ __forceinline__ bool mbar_try_wait(uint64_t* b, uint32_t parity) {  // W8 ptx.cuh:100-108, DG ptx/tma.cuh:70
  uint32_t ok = 0;
  K7_ASM("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 0x989680;\n selp.u32 %0, 1, 0, p;\n}"
         : "=r"(ok)
         : "r"(smem_u32(b)), "r"(parity)
         : "memory");
  return ok != 0;
}
// A load the assembler may not drop or cache: used only to spend time between two tries of a bounded wait.
__device__ __forceinline__ uint32_t ld_volatile_global(const uint32_t* ptr) {  // DG ptx/ld_st.cuh:241-245
  uint32_t v = 0;
  K7_ASM("ld.volatile.global.b32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

// ------------------------------------------------------------------ proxies, TMA
// plain (generic-proxy) stores to shared memory -> visible to the async proxy (TMA, tcgen05.cp, tcgen05.mma)
__device__ __forceinline__ void fence_proxy_async_smem() {  // W8 ptx.cuh:32, W3 ptx.cuh:84-86
  K7_ASM("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {  // W8 ptx.cuh:170-174
  uint64_t p = 0;
  K7_ASM("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// 2D tiled TMA load, coordinates in elements (c0 innermost), into this CTA's shared memory.
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint64_t* bar,
                                            uint64_t policy) {  // W8 ptx.cuh:123-130
  K7_ASM("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], "
         "%5;" ::"r"(smem_u32(dst)),
         "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
         : "memory");
}

// ------------------------------------------------------------------ tensor memory
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t ncols) {  // W8 umma.cuh:9-13 (whole warp)
  K7_ASM("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(smem_dst)), "r"(ncols)
         : "memory");
  K7_ASM("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {  // W8 umma.cuh:14-16 (whole warp)
  K7_ASM("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() {  // W8 umma.cuh:17, DG ptx/tcgen05.cuh:211-213
  K7_ASM("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {  // W8 umma.cuh:18, DG ptx/tcgen05.cuh:215-217
  K7_ASM("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// Arrive on an mbarrier once all earlier tcgen05.cp / tcgen05.mma of this thread have completed.
__device__ __forceinline__ void tc_commit(uint64_t* bar) {  // W8 umma.cuh:65-68
  K7_ASM("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_wait_ld() {  // W8 umma.cuh:136
  K7_ASM("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {  // W1 ptx.cuh:103
  K7_ASM("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// 32 lanes x 32 bit, 8 or 16 columns: lane t of the warp reads / writes tensor-memory lane (taddr.lane + t).
__device__ __forceinline__ void tc_ld8(uint32_t taddr, uint32_t (&r)[8]) {  // W8 umma.cuh:110-115
  K7_ASM("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
         : "r"(taddr));
}
__device__ __forceinline__ void tc_ld16(uint32_t taddr, uint32_t (&r)[16]) {  // W8 umma.cuh:116-124
  K7_ASM("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
         "[%16];"
         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
           "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
         : "r"(taddr));
}
__device__ __forceinline__ void tc_st8(uint32_t taddr, const uint32_t (&r)[8]) {  // W1 ptx.cuh:125-130
  K7_ASM("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr), "r"(r[0]), "r"(r[1]),
         "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
         : "memory");
}

// ------------------------------------------------------------------ descriptors (layouts: W3 ptx.cuh:123-139)
// Shared-memory matrix descriptor, SM100 version 1 (bit fields also in PTX 9.7.18.4.1, table 49).
__device__ __forceinline__ uint64_t smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {  // W3 ptx.cuh:124-132
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;  // version = 1
  d |= static_cast<uint64_t>(layout & 7) << 61;
  return d;
}
constexpr uint32_t kLayoutNone = 0;   // W3 ptx.cuh:134
constexpr uint32_t kLayoutSw128 = 2;  // W3 ptx.cuh:133
// Scale-factor source of the copy: rows of 16 B, no swizzle, 8 rows per 128 B (W8 umma.cuh:32-40,
// W3 mxf8_core.cuh:59, DG mma/sm100.cuh:43-49).
__device__ __forceinline__ uint64_t sdesc_sf(const void* p) { return smem_desc(smem_u32(p), 0, 128, kLayoutNone); }
// Instruction descriptor, kind::mxf8f6f4 block-scaled, E4M3 x E4M3 -> F32, UE8M0 scales, both K-major.
__host__ __device__ constexpr uint32_t idesc_mxf8(uint32_t m, uint32_t n, uint32_t sfa_id, uint32_t sfb_id) {  // W3 ptx.cuh:137-139
  return (sfb_id << 4) | ((n >> 3) << 17) | (1u << 23) | ((m >> 4) << 24) | (sfa_id << 29);
}

// ------------------------------------------------------------------ copies and the multiply (one thread issues)
__device__ __forceinline__ void cp_plain_32x128b_x4(uint32_t taddr, uint64_t sdesc) {  // W8 umma.cuh:70-72
  K7_ASM("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void cp_under_test(uint32_t taddr, uint64_t sdesc) {  // see K7_CP_FORM above
  K7_ASM(K7_CP_STR ::"r"(taddr), "l"(sdesc) : "memory");
}
// D[tmem_d] (+)= A[adesc] x B[bdesc] with block scales read from tensor memory.
__device__ __forceinline__ void mma_mxf8(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t tmem_sfa,
                                         uint32_t tmem_sfb, uint32_t accumulate) {  // W8 umma.cuh:56-63, DG ptx/tcgen05.cuh:55-63
  K7_ASM("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
         " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(tmem_d),
         "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
         : "memory");
}

}  // namespace k7
