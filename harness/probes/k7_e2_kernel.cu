// k7_e2_kernel.cu -- K7 experiment E2: what does a packed scale column look like in shared memory after a TMA
// load, and in tensor memory after tcgen05.cp? One small kernel (1 CTA, 128 threads) plus a plain C entry point for
// ctypes. BOX ONLY: needs a GB300 (sm_103a). Built by k7_e2_probe.py the way Garner's kernels are built
// (kernels/w8-persist/build.sh:10-12): nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared
// -Xcompiler -fPIC, once per K7_CP_FORM.
//
// What one launch does, in order (every mbarrier wait is bounded, so a wrong byte count cannot hang the box):
//   0. copies the host's 24576-byte "image" into shared memory with plain stores;
//   1. control: tcgen05.st a known pattern into tensor-memory columns 0..7, tcgen05.ld it back      -> out_ctl
//   2. source: nothing more (manual image), or ONE TMA load of the box into shared memory offset 0  -> out_status[0..1]
//   3. dumps shared memory [0, 2048)                                                                -> out_smem
//   4. tcgen05.st a sentinel into columns 0..7, then ONE tcgen05.cp (the form under test) from shared memory
//      offset 0 into columns 0..3, then tcgen05.ld columns 0..7                                     -> out_tmem
//   5. optional: one block-scaled multiply, 128 x 16 x 128 (4 K blocks), weight scales = columns 0..3 as step 4
//      left them, activation scales = a plain copy into columns 4..7, accumulator read back         -> out_acc
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "k7_e2_ptx.cuh"

#ifndef K7_EXTRA_ENUMS
#define K7_EXTRA_ENUMS 1
#endif

namespace k7 {

// Shared-memory layout. k7_e2_probe.py builds the image with the same numbers and checks them through k7e2_abi().
constexpr uint32_t kSfOff = 0;            // the scale source under test: 2048 B = up to 128 rows of 16 B
constexpr uint32_t kSfBytes = 2048;
constexpr uint32_t kSfbOff = 2048;        // activation scales, one 512 B atom, plain bytes (multiply only)
constexpr uint32_t kBOff = 4096;          // activations: 16 rows x 128 B, K-major, SWIZZLE_128B (multiply only)
constexpr uint32_t kAOff = 8192;          // weights: 128 rows x 128 B, K-major, SWIZZLE_128B (multiply only)
constexpr uint32_t kImageBytes = 24576;
constexpr uint32_t kBarOff = kImageBytes; // two mbarriers and the tensor-memory base word
constexpr uint32_t kSmemBytes = kImageBytes + 1024;
constexpr uint32_t kThreads = 128;        // 4 warps: warp w owns tensor-memory lanes 32w .. 32w+31 (PTX 9.7.18.8.1)
constexpr uint32_t kTmemCols = 32;        // smallest allocation: 32 columns (PTX 9.7.18.1.2)
constexpr uint32_t kColSf = 0;            // columns 0..3: destination of the copy under test
constexpr uint32_t kColSfb = 4;           // columns 4..7: sentinel in step 4, activation scales in step 5
constexpr uint32_t kColAcc = 8;           // columns 8..23: 128 x 16 fp32 accumulator
constexpr uint32_t kMmaM = 128, kMmaN = 16;  // N % 16 == 0 at M = 128 (DG impls/sm100_fp8_fp4_mega_moe.cuh:1058-1061)
constexpr uint32_t kWaitIters = 400;      // tries per bounded wait; each try may block up to the 10 ms hint
constexpr uint32_t kSpinLoads = 256;      // volatile loads between two tries (time the assembler cannot remove)
constexpr uint32_t kSentinel = 0xEEEEEEEEu;
constexpr uint32_t kDoneMagic = 0x4B374532u;  // "K7E2"

struct Params {
  CUtensorMap tmap;      // by value in the kernel parameters (W3 stage_wq_b.cuh:12, W8 woa_kernel.cuh:87); the type
                         // carries its own alignment in cuda.h, so no alignas here
  const uint8_t* image;
  uint32_t* out_ctl;     // [128][8]
  uint32_t* out_smem;    // [kSfBytes / 4]
  uint32_t* out_tmem;    // [128][8]
  uint32_t* out_acc;     // [128][16]
  uint32_t* out_status;  // [16]
  int32_t src_mode;      // 0: the image is the source, 1: TMA load
  int32_t c0, c1;        // TMA coordinates (elements, rows)
  uint32_t expect_tx;    // bytes the TMA barrier is told to expect
  int32_t do_mma;
};
static_assert(alignof(CUtensorMap) >= 64, "cuda.h's CUtensorMap must be 64-byte aligned (cuTensorMapEncodeTiled requires it)");

// Wait for an mbarrier phase, at most kWaitIters tries. Returns 1 if it completed. Between two tries it does
// kSpinLoads volatile loads of `spin` (a device word nobody writes), so the wait still lasts long enough for a
// copy to finish if try_wait should return at once instead of blocking for its hint.
__device__ __forceinline__ uint32_t wait_bar(uint64_t* bar, uint32_t parity, const uint32_t* spin, uint32_t& iters) {
#pragma unroll 1
  for (iters = 0; iters < kWaitIters; ++iters) {
    if (mbar_try_wait(bar, parity)) return 1u;
#pragma unroll 1
    for (uint32_t j = 0; j < kSpinLoads; ++j) (void)ld_volatile_global(spin);
  }
  return 0u;
}

__global__ void __launch_bounds__(kThreads, 1) k7e2_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* bar_tma = reinterpret_cast<uint64_t*>(smem + kBarOff);
  uint64_t* bar_tc = bar_tma + 1;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bar_tma + 2);
  const uint32_t tid = threadIdx.x, w = tid >> 5, row = tid;
  const uint32_t* spin = p.out_status + 15;  // never written by the kernel

  // ---- step 0: barriers, tensor memory, image -> shared memory (the order of W3 standalone.cuh:46-57)
  if (tid == 0) {
    mbar_init(bar_tma, 1);
    mbar_init(bar_tc, 1);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(tmem_slot, kTmemCols);
  for (uint32_t i = tid; i < kImageBytes / 16; i += kThreads)
    *reinterpret_cast<uint4*>(smem + 16 * i) = *reinterpret_cast<const uint4*>(p.image + 16 * i);
  fence_proxy_async_smem();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;
  const uint32_t tq = tmem + ((w * 32u) << 16);  // this warp's 32 lanes (W3 mxf8_core.cuh:96)
  uint32_t r[8], g[8];

  // ---- step 1: control of the readback path, tcgen05.st -> tcgen05.ld
#pragma unroll
  for (uint32_t c = 0; c < 8; ++c) r[c] = 0xC0DE0000u | (row << 8) | c;
  tc_st8(tq + kColSf, r);
  tc_wait_st();
  tc_ld8(tq + kColSf, g);
  tc_wait_ld();
#pragma unroll
  for (uint32_t c = 0; c < 8; ++c) p.out_ctl[row * 8 + c] = g[c];

  // ---- step 2: the source. One TMA box into shared memory offset 0 (the image holds a sentinel there).
  if (p.src_mode == 1) {
    if (tid == 0) {
      mbar_arrive_expect_tx(bar_tma, p.expect_tx);
      tma_load_2d(smem + kSfOff, &p.tmap, p.c0, p.c1, bar_tma, policy_evict_first());
      uint32_t it = 0;
      const uint32_t ok = wait_bar(bar_tma, 0, spin, it);
      p.out_status[0] = ok;
      p.out_status[1] = it;
    }
    __syncthreads();
  }

  // ---- step 3: shared-memory dump
  for (uint32_t i = tid; i < kSfBytes / 4; i += kThreads)
    p.out_smem[i] = *reinterpret_cast<const uint32_t*>(smem + kSfOff + 4 * i);

  // ---- step 4: sentinel, the copy under test, readback
#pragma unroll
  for (uint32_t c = 0; c < 8; ++c) r[c] = kSentinel;
  tc_st8(tq + kColSf, r);
  tc_wait_st();
  tc_fence_before();
  __syncthreads();
  if (tid == 0) {
    tc_fence_after();
    cp_under_test(tmem + kColSf, sdesc_sf(smem + kSfOff));
    tc_commit(bar_tc);
  }
  uint32_t it_cp = 0;
  const uint32_t ok_cp = wait_bar(bar_tc, 0, spin, it_cp);
  tc_fence_after();
  tc_ld8(tq + kColSf, g);
  tc_wait_ld();
#pragma unroll
  for (uint32_t c = 0; c < 8; ++c) p.out_tmem[row * 8 + c] = g[c];
  if (tid == 0) {
    p.out_status[2] = ok_cp;
    p.out_status[3] = it_cp;
  }

  // ---- step 5: one block-scaled multiply over K = 128 (the call shape of W3 mxf8_core.cuh:59-66)
  if (p.do_mma != 0) {
    tc_fence_before();
    __syncthreads();
    if (tid == 0) {
      tc_fence_after();
      cp_plain_32x128b_x4(tmem + kColSfb, sdesc_sf(smem + kSfbOff));
      const uint32_t a0 = smem_u32(smem + kAOff), b0 = smem_u32(smem + kBOff);
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k)
        mma_mxf8(tmem + kColAcc, smem_desc(a0 + k * 32, 0, 1024, kLayoutSw128), smem_desc(b0 + k * 32, 0, 1024, kLayoutSw128),
                 idesc_mxf8(kMmaM, kMmaN, k, k), tmem + kColSf, tmem + kColSfb, k != 0 ? 1u : 0u);
      tc_commit(bar_tc);
    }
    uint32_t it_m = 0;
    const uint32_t ok_m = wait_bar(bar_tc, 1, spin, it_m);
    tc_fence_after();
    uint32_t a[16];
    tc_ld16(tq + kColAcc, a);
    tc_wait_ld();
#pragma unroll
    for (uint32_t c = 0; c < 16; ++c) p.out_acc[row * 16 + c] = a[c];
    if (tid == 0) {
      p.out_status[4] = ok_m;
      p.out_status[5] = it_m;
    }
  }

  // ---- end: release tensor memory (W3 standalone.cuh:85-88)
  tc_fence_before();
  __syncthreads();
  if (w == 0) tmem_dealloc(tmem, kTmemCols);
  if (tid == 0) {
    p.out_status[6] = tmem;
    p.out_status[7] = kDoneMagic;
  }
}

// cuTensorMapEncodeTiled through the runtime's driver entry point (W3 host_tmap.h:10-24, W8 oproj/w8oproj_lib.cu:38-48).
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                   const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_fn() {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q;
  if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q) != cudaSuccess ||
      q != cudaDriverEntryPointSuccess)
    return nullptr;
  return reinterpret_cast<EncodeTiledFn>(fn);
}

struct DevBuf {
  void* p = nullptr;
  ~DevBuf() {
    if (p != nullptr) cudaFree(p);
  }
};

}  // namespace k7

extern "C" {

// Every field is 8 bytes, so the ctypes mirror cannot disagree about padding (k7e2_abi(0) checks the size).
struct K7Req {
  const unsigned char* image;  // k7::kImageBytes host bytes
  const unsigned char* src;    // host bytes of the global tensor the TMA reads (src_mode 1)
  unsigned int* out_ctl;       // [128 * 8]
  unsigned int* out_smem;      // [kSfBytes / 4]
  unsigned int* out_tmem;      // [128 * 8]
  unsigned int* out_acc;       // [128 * 16]
  unsigned int* out_status;    // [16]
  long long src_bytes;
  long long src_mode;          // 0 manual, 1 TMA
  long long tma_dtype;         // index for k7e2_enum()
  long long tma_swizzle;       // 0 none, 1 128 B
  long long dim0, dim1;        // global tensor: elements per row, rows
  long long stride1;           // bytes per row
  long long box0, box1;        // box: elements, rows
  long long c0, c1;            // box origin: element, row
  long long expect_tx;
  long long do_mma;
  long long cuda_err;          // out: cudaError_t of the call that failed
  long long encode_err;        // out: CUresult of cuTensorMapEncodeTiled (-1 no entry point, -2 unknown data type)
};

// 0 sizeof(K7Req), 1 image bytes, 2 scale-source bytes, 3 copy form, 4 rows the copy form reads, 5 extra enum names
// compiled in, 6..9 the shared-memory offsets (scale source, activation scales, activations, weights)
long long k7e2_abi(int which) {
  switch (which) {
    case 0: return sizeof(K7Req);
    case 1: return k7::kImageBytes;
    case 2: return k7::kSfBytes;
    case 3: return K7_CP_FORM;
    case 4: return K7_CP_ROWS;
    case 5: return K7_EXTRA_ENUMS;
    case 6: return k7::kSfOff;
    case 7: return k7::kSfbOff;
    case 8: return k7::kBOff;
    case 9: return k7::kAOff;
    default: return -1;
  }
}

const char* k7e2_cp_string() { return K7_CP_STR; }

// The tensor-map data types by name; the names are resolved by the compiler against the image's cuda.h.
// 0 UINT8 (W3 host_tmap.h:34), 2 16U4_ALIGN16B (SGLang moe_mxfp4.cuh:1145); 1 and 3 only with K7_EXTRA_ENUMS.
long long k7e2_enum(int which) {
  switch (which) {
    case 0: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case 2: return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
#if K7_EXTRA_ENUMS
    case 1: return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    case 3: return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
#endif
    default: return -1;
  }
}

const char* k7e2_err_string(long long cuda_err) { return cudaGetErrorString(static_cast<cudaError_t>(cuda_err)); }

// Device 0: name into `name` (n bytes), compute capability, total memory. Returns the cudaError_t.
long long k7e2_device(char* name, long long n, long long* major, long long* minor, long long* total_bytes) {
  cudaDeviceProp prop;
  const cudaError_t e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) return e;
  if (name != nullptr && n > 0) {
    std::strncpy(name, prop.name, static_cast<size_t>(n) - 1);
    name[n - 1] = 0;
  }
  if (major != nullptr) *major = prop.major;
  if (minor != nullptr) *minor = prop.minor;
  if (total_bytes != nullptr) *total_bytes = static_cast<long long>(prop.totalGlobalMem);
  return 0;
}

// 0 ok; 1 bad argument; 2 tensor map (encode_err); 3 device allocation; 4 launch; 5 kernel failed (cuda_err from the
// synchronize); 6 copy.
long long k7e2_run(K7Req* q) {
  if (q == nullptr) return 1;
  q->cuda_err = 0;
  q->encode_err = 0;
  if (q->image == nullptr || q->out_ctl == nullptr || q->out_smem == nullptr || q->out_tmem == nullptr ||
      q->out_acc == nullptr || q->out_status == nullptr)
    return 1;
  constexpr size_t kSrcCap = 8192;
  constexpr size_t nCtl = 128 * 8 * 4, nSmem = k7::kSfBytes, nTmem = 128 * 8 * 4, nAcc = 128 * 16 * 4, nStatus = 16 * 4;
  if (q->src_mode == 1 && (q->src == nullptr || q->src_bytes <= 0 || q->src_bytes > static_cast<long long>(kSrcCap))) return 1;
  cudaError_t e = cudaSuccess;
#define K7_TRY(call, code)        \
  do {                            \
    e = (call);                   \
    if (e != cudaSuccess) {       \
      q->cuda_err = e;            \
      return code;                \
    }                             \
  } while (0)
  k7::DevBuf image, src, ctl, sm, tm, acc, status;
  K7_TRY(cudaMalloc(&image.p, k7::kImageBytes), 3);
  K7_TRY(cudaMalloc(&src.p, kSrcCap), 3);
  K7_TRY(cudaMalloc(&ctl.p, nCtl), 3);
  K7_TRY(cudaMalloc(&sm.p, nSmem), 3);
  K7_TRY(cudaMalloc(&tm.p, nTmem), 3);
  K7_TRY(cudaMalloc(&acc.p, nAcc), 3);
  K7_TRY(cudaMalloc(&status.p, nStatus), 3);
  K7_TRY(cudaMemcpy(image.p, q->image, k7::kImageBytes, cudaMemcpyHostToDevice), 6);
  K7_TRY(cudaMemset(src.p, 0, kSrcCap), 6);
  K7_TRY(cudaMemset(ctl.p, 0xA5, nCtl), 6);  // 0xA5A5A5A5 = "the kernel never wrote this word"
  K7_TRY(cudaMemset(sm.p, 0xA5, nSmem), 6);
  K7_TRY(cudaMemset(tm.p, 0xA5, nTmem), 6);
  K7_TRY(cudaMemset(acc.p, 0xA5, nAcc), 6);
  K7_TRY(cudaMemset(status.p, 0xA5, nStatus), 6);

  k7::Params p;
  std::memset(&p, 0, sizeof(p));
  if (q->src_mode == 1) {
    K7_TRY(cudaMemcpy(src.p, q->src, static_cast<size_t>(q->src_bytes), cudaMemcpyHostToDevice), 6);
    const k7::EncodeTiledFn enc = k7::encode_fn();
    if (enc == nullptr) {
      q->encode_err = -1;
      return 2;
    }
    const long long dt = k7e2_enum(static_cast<int>(q->tma_dtype));
    if (dt < 0) {
      q->encode_err = -2;
      return 2;
    }
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(q->dim0), static_cast<cuuint64_t>(q->dim1)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(q->stride1)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(q->box0), static_cast<cuuint32_t>(q->box1)};
    const cuuint32_t estr[2] = {1, 1};
    const CUresult r = enc(&p.tmap, static_cast<CUtensorMapDataType>(dt), 2, src.p, dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           q->tma_swizzle != 0 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
      q->encode_err = static_cast<long long>(r);
      return 2;
    }
  }
  p.image = static_cast<const uint8_t*>(image.p);
  p.out_ctl = static_cast<uint32_t*>(ctl.p);
  p.out_smem = static_cast<uint32_t*>(sm.p);
  p.out_tmem = static_cast<uint32_t*>(tm.p);
  p.out_acc = static_cast<uint32_t*>(acc.p);
  p.out_status = static_cast<uint32_t*>(status.p);
  p.src_mode = static_cast<int32_t>(q->src_mode);
  p.c0 = static_cast<int32_t>(q->c0);
  p.c1 = static_cast<int32_t>(q->c1);
  p.expect_tx = static_cast<uint32_t>(q->expect_tx);
  p.do_mma = static_cast<int32_t>(q->do_mma);

  K7_TRY(cudaFuncSetAttribute(k7::k7e2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k7::kSmemBytes), 4);
  k7::k7e2_kernel<<<1, k7::kThreads, k7::kSmemBytes>>>(p);
  K7_TRY(cudaGetLastError(), 4);
  e = cudaDeviceSynchronize();
  const cudaError_t sync_err = e;
  // copy back whatever was written, even after a failed kernel (the copies fail too if the context is gone)
  cudaMemcpy(q->out_status, status.p, nStatus, cudaMemcpyDeviceToHost);
  cudaMemcpy(q->out_ctl, ctl.p, nCtl, cudaMemcpyDeviceToHost);
  cudaMemcpy(q->out_smem, sm.p, nSmem, cudaMemcpyDeviceToHost);
  cudaMemcpy(q->out_tmem, tm.p, nTmem, cudaMemcpyDeviceToHost);
  cudaMemcpy(q->out_acc, acc.p, nAcc, cudaMemcpyDeviceToHost);
  if (sync_err != cudaSuccess) {
    q->cuda_err = sync_err;
    return 5;
  }
  K7_TRY(cudaGetLastError(), 6);
#undef K7_TRY
  return 0;
}

}  // extern "C"
