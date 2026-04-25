#include "kernel.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

// [B200 optimization — Phase 1b] PDL (Programmatic Dependent Launch) wrapper.
// Lets the next kernel start its prologue (instruction fetch, argument load,
// SMEM allocation) while the current kernel is still finishing its tail. On
// SM90+ this overlaps ~3-5µs of launch latency per kernel transition. Used by
// the producer kernels in our pipeline so the downstream consumer doesn't wait
// on a fully-drained dependency.
//
// Usage: replace  kernel<<<grid, block, smem, stream>>>(args...)  with
//                 LAUNCH_PDL(kernel, grid, block, smem, stream, args...)
// Caller is responsible for the consumer-side `cudaGridDependencySynchronize()`
// if it needs to read results from the producer — our consumers are other
// CUTLASS or CUDA kernels that PDL-wait implicitly.
#define LAUNCH_PDL(kernel_fn, grid, block, smem, stream, ...)                  \
    do {                                                                       \
        cudaLaunchConfig_t _cfg = {};                                          \
        _cfg.gridDim = (grid); _cfg.blockDim = (block);                        \
        _cfg.dynamicSmemBytes = (smem); _cfg.stream = (stream);                \
        cudaLaunchAttribute _attr[1] = {};                                     \
        _attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;      \
        _attr[0].val.programmaticStreamSerializationAllowed = 1;               \
        _cfg.attrs = _attr; _cfg.numAttrs = 1;                                 \
        CUDA_CHECK(cudaLaunchKernelEx(&_cfg, kernel_fn, ##__VA_ARGS__));       \
    } while (0)

// Helper to build a 3D grid that handles Tk > 65535 (CUDA grid.y limit)
static inline dim3 make_row_grid(int col_blocks, int Tk) {
    if (Tk <= 65535) return dim3(col_blocks, Tk, 1);
    // Split Tk across y and z: pick z so y*z >= Tk
    int z = (Tk + 65534) / 65535;
    int y = (Tk + z - 1) / z;
    return dim3(col_blocks, y, z);
}

// Warp reduce max
__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffffu, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

// 1) No-aux routing kernel
// One block per token (T), 8 warps per block (256 threads), one warp per group
__global__ void noaux_routing_topk8_kernel(
    const float* __restrict__ logits,   // [T, 256]
    const float* __restrict__ bias,     // [256]
    int T,
    float routed_scaling_factor,
    int* __restrict__ topk_idx,         // [T, 8]
    float* __restrict__ topk_w) {       // [T, 8]

  __shared__ float group_scores[ROUTE_NUM_GROUP]; // 8
  __shared__ unsigned int keep_group_mask;        // bitmask of 8 groups
  __shared__ float warpCandVal[ROUTE_NUM_GROUP * ROUTE_TOP_K];        // 8*8 = 64
  __shared__ int   warpCandIdx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
  __shared__ float warpCandSNoBias[ROUTE_NUM_GROUP * ROUTE_TOP_K];

  int t = blockIdx.x;
  if (t >= T) return;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5; // 0..7
  const int e = warp * ROUTE_GROUP_SIZE + lane;  // expert index 0..255

  // Load and compute s and s_with_bias
  float l = logits[t * NUM_EXPERTS_GLOBAL + e];
  float s = 1.f / (1.f + __expf(-l));
  float sb = s + bias[e];

  // Compute group top-2 sum within warp
  float v = sb;
  float m1 = warp_max(v);
  unsigned mask1 = __ballot_sync(0xffffffffu, v == m1);
  int idx1_lane = __ffs(mask1) - 1;
  float v2 = (lane == idx1_lane) ? -CUDART_INF_F : v;
  float m2 = warp_max(v2);
  if (lane == 0) {
    group_scores[warp] = m1 + m2;
  }
  __syncthreads();

  // Select top-4 groups on a single thread
  if (threadIdx.x == 0) {
    float temp_scores[ROUTE_NUM_GROUP];
    #pragma unroll
    for (int g = 0; g < ROUTE_NUM_GROUP; ++g) temp_scores[g] = group_scores[g];
    unsigned int mask_bits = 0u;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOPK_GROUP; ++j) {
      int best = 0;
      float bestv = temp_scores[0];
      #pragma unroll
      for (int g = 1; g < ROUTE_NUM_GROUP; ++g) {
        if (temp_scores[g] > bestv) { bestv = temp_scores[g]; best = g; }
      }
      mask_bits |= (1u << best);
      temp_scores[best] = -CUDART_INF_F;
    }
    keep_group_mask = mask_bits;
  }
  __syncthreads();

  // Prune unkept groups by setting -inf, keep sb for kept groups
  bool keep = ((keep_group_mask >> warp) & 1u) != 0u;
  float cur = keep ? sb : -CUDART_INF_F;

  // Compute top-8 within this warp (group)
  #pragma unroll
  for (int j = 0; j < ROUTE_TOP_K; ++j) {
    float m = warp_max(cur);
    unsigned msk = __ballot_sync(0xffffffffu, cur == m);
    int max_lane = __ffs(msk) - 1;
    float s_no_bias_sel = __shfl_sync(0xffffffffu, s, max_lane);
    if (lane == 0) {
      int base = warp * ROUTE_TOP_K + j;
      warpCandVal[base] = m;
      warpCandIdx[base] = warp * ROUTE_GROUP_SIZE + max_lane;
      warpCandSNoBias[base] = s_no_bias_sel;
    }
    if (lane == max_lane) cur = -CUDART_INF_F;
  }
  __syncthreads();

  // Merge 64 candidates → top-8 globally using WARP-PARALLEL top-k.
  // Previously serialized on thread 0 (~512 sequential ops). Warp-parallel
  // version: each lane holds 2 candidates, 8 iterations of warp_max.
  // ~6× faster on the merge step.
  if (warp == 0) {
    // 64 candidates / 32 lanes = 2 per lane.
    float v0 = warpCandVal[lane * 2 + 0];
    float v1 = warpCandVal[lane * 2 + 1];
    int   i0 = warpCandIdx[lane * 2 + 0];
    int   i1 = warpCandIdx[lane * 2 + 1];
    float s0 = warpCandSNoBias[lane * 2 + 0];
    float s1 = warpCandSNoBias[lane * 2 + 1];

    __shared__ int   sel_idx_sh[ROUTE_TOP_K];
    __shared__ float sel_s_sh  [ROUTE_TOP_K];

    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      // Each lane's "current best" between its 2 candidates.
      bool pick0 = (v0 >= v1);
      float v = pick0 ? v0 : v1;
      // Warp-wide max → global value across 64 candidates.
      float gmax = warp_max(v);
      // Find which lane holds the max (ties broken by lowest lane).
      unsigned msk = __ballot_sync(0xffffffffu, v == gmax);
      int winner = __ffs(msk) - 1;
      // Read the winner's index and s_no_bias.
      int   sel_i = pick0 ? i0 : i1;
      float sel_s = pick0 ? s0 : s1;
      int   sel_i_bcast = __shfl_sync(0xffffffffu, sel_i,  winner);
      float sel_s_bcast = __shfl_sync(0xffffffffu, sel_s,  winner);
      if (lane == 0) {
        sel_idx_sh[j] = sel_i_bcast;
        sel_s_sh[j]   = sel_s_bcast;
      }
      // Winner lane: mark its used candidate as consumed.
      if (lane == winner) {
        if (pick0) v0 = -CUDART_INF_F;
        else       v1 = -CUDART_INF_F;
      }
    }

    // Normalize and write — single thread.
    if (lane == 0) {
      float sumw = 0.f;
      #pragma unroll
      for (int j = 0; j < ROUTE_TOP_K; ++j) sumw += sel_s_sh[j];
      sumw = fmaxf(sumw, 1e-20f);
      #pragma unroll
      for (int j = 0; j < ROUTE_TOP_K; ++j) {
        float w = (sel_s_sh[j] / sumw) * routed_scaling_factor;
        topk_idx[t * ROUTE_TOP_K + j] = sel_idx_sh[j];
        topk_w  [t * ROUTE_TOP_K + j] = w;
      }
    }
  }
}

// Fused SFA preparation: replaces 4 torch ops (cast int32→int64, index_select,
// contiguous, mul_) with a single kernel. Output: sfa[kb, i] = hs_scale[kb, safe_ids[i]] * valid_mask[i]
__global__ void prepare_sfa_kernel(
    const float* __restrict__ hs_scale,   // [K_blocks, T] row-major
    const int*   __restrict__ safe_ids,   // [padded_total]
    const float* __restrict__ valid_mask, // [padded_total]
    int T, int padded_total, int K_blocks,
    float*       __restrict__ sfa)        // [K_blocks, padded_total] row-major
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int kb = blockIdx.y;
    if (i >= padded_total || kb >= K_blocks) return;
    int sid = safe_ids[i];
    float v = valid_mask[i];
    float s = hs_scale[(size_t)kb * T + sid];
    sfa[(size_t)kb * padded_total + i] = s * v;
}

void launch_prepare_sfa(
    const float* hs_scale, const int* safe_ids, const float* valid_mask,
    int T, int padded_total, int K_blocks,
    float* sfa, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((padded_total + block.x - 1) / block.x, K_blocks);
    prepare_sfa_kernel<<<grid, block, 0, stream>>>(
        hs_scale, safe_ids, valid_mask, T, padded_total, K_blocks, sfa);
    CUDA_CHECK(cudaGetLastError());
}

// 2) Hidden block scale application (in-place)
__global__ void apply_hidden_block_scale_kernel(
    float* __restrict__ A,            // [T, H]
    const float* __restrict__ S,      // [H/128, T] in row-major
    int T, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = T * H;
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    int t = i / H;
    int h = i - t * H;
    int hb = h >> 7; // h/128
    float sc = S[hb * T + t];
    A[i] *= sc;
  }
}

// 3) Apply 128x128 block scale to 2D matrix (in-place)
__global__ void apply_block_scale_128x128_kernel(
    float* __restrict__ M,     // [rows, cols]
    int rows, int cols,
    const float* __restrict__ S,// [rows/128, cols/128]
    int Sb_rows, int Sb_cols) {

  int blk_row = blockIdx.y; // 0..rows/128 - 1
  int blk_col = blockIdx.x; // 0..cols/128 - 1
  float scale = S[blk_row * Sb_cols + blk_col];

  int row_base = blk_row * BLOCK_SIZE_128;
  int col_base = blk_col * BLOCK_SIZE_128;

  int tx = threadIdx.x; // 0..31
  int ty = threadIdx.y; // 0..7

  // Fully cover the 128x128 tile using 32x8 threads
  for (int r = ty; r < BLOCK_SIZE_128; r += blockDim.y) {
    int row = row_base + r;
    float* row_ptr = M + row * cols;
    for (int c = tx; c < BLOCK_SIZE_128; c += blockDim.x) {
      int col = col_base + c;
      row_ptr[col] *= scale;
    }
  }
}

// 4) Count assignments per local expert
__global__ void count_local_assignments_kernel(
    const int* __restrict__ topk_idx,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ counts) {         // [32]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int ge = topk_idx[base + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      atomicAdd(&counts[le], 1);
    }
  }
}

// 5) Fill assignments using prefix offsets
__global__ void fill_local_assignments_kernel(
    const int* __restrict__ topk_idx,   // [T, 8]
    const float* __restrict__ topk_w,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ offsets_inout,    // [32], running counters
    int* __restrict__ token_ids_out,    // [total]
    float* __restrict__ token_w_out) {  // [total]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int ge = topk_idx[base + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      int pos = atomicAdd(&offsets_inout[le], 1);
      token_ids_out[pos] = t;
      token_w_out[pos] = topk_w[base + k];
    }
  }
}

// Variant that ALSO emits inverse map (token, k) → padded_slot for finalize kernel.
// Eliminates the need for atomic scatter at the end of the pipeline.
__global__ void fill_local_assignments_with_inverse_kernel(
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    int T,
    int local_expert_offset,
    int* __restrict__ offsets_inout,         // [32], running unpadded counters
    const int* __restrict__ unpadded_offsets,// [33] read-only base offsets
    const int* __restrict__ padded_offsets,  // [33] read-only base offsets
    int* __restrict__ token_ids_out,         // [total_assign]
    float* __restrict__ token_w_out,         // [total_assign]
    int* __restrict__ inv_padded_slot)       // [T, topK]  -1 if non-local
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int ge = topk_idx[base + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      int unp_pos = atomicAdd(&offsets_inout[le], 1);
      token_ids_out[unp_pos] = t;
      token_w_out[unp_pos] = topk_w[base + k];
      // Map unpadded → padded: padded_pos = padded_off[le] + (unp_pos - unp_off[le])
      int padded_pos = padded_offsets[le] + (unp_pos - unpadded_offsets[le]);
      inv_padded_slot[base + k] = padded_pos;
    } else {
      inv_padded_slot[base + k] = -1;
    }
  }
}

// 6) Gather rows [T,H] -> [Tk,H]
__global__ void gather_rows_kernel(
    const float* __restrict__ A,     // [T, H]
    const int* __restrict__ token_ids,// [Tk]
    int /*T*/, int Tk, int H,
    float* __restrict__ A_out) {     // [Tk, H]
  int row = blockIdx.z * gridDim.y + blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  int t = token_ids[row];
  A_out[row * H + col] = A[t * H + col];
}

// 7) SwiGLU kernel
__global__ void swiglu_kernel(
    const float* __restrict__ G1, // [Tk, 4096]
    int Tk,
    float* __restrict__ C) {      // [Tk, 2048]
  int row = blockIdx.z * gridDim.y + blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= INTERMEDIATE_SIZE) return;
  const float* g1_row = G1 + row * GEMM1_OUT_SIZE;
  float x1 = g1_row[col];
  float x2 = g1_row[col + INTERMEDIATE_SIZE];
  float silu = x2 / (1.0f + __expf(-x2));
  C[row * INTERMEDIATE_SIZE + col] = silu * x1;
}

// 8) Accumulate O into output with weights (non-atomic, used per-expert)
__global__ void accumulate_weighted_add_kernel(
    const float* __restrict__ O,       // [Tk, H]
    const int* __restrict__ token_ids, // [Tk]
    const float* __restrict__ weights, // [Tk]
    int Tk, int H,
    float* __restrict__ output) {      // [T, H]
  int row = blockIdx.z * gridDim.y + blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  int t = token_ids[row];
  float w = weights[row];
  float val = O[row * H + col] * w;
  output[t * H + col] += val;
}

// 8b) Fused atomic accumulate — safe across experts sharing same token.
// Use T_max as sentinel: padding rows with token_ids[i]==T_max are skipped.
__global__ void accumulate_weighted_add_atomic_kernel(
    const float* __restrict__ O,       // [N, H] — N = padded_total
    const int* __restrict__ token_ids, // [N]  (T_max = padding sentinel)
    const float* __restrict__ weights, // [N]  (0.0 for padding)
    int N, int H, int T_max,
    float* __restrict__ output) {      // [T_max, H]
  int row = blockIdx.z * gridDim.y + blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= H) return;
  int t = token_ids[row];
  if (t < 0 || t >= T_max) return;   // skip padding
  float w = weights[row];
  if (w == 0.0f) return;              // skip zero-weight
  float val = O[row * H + col] * w;
  atomicAdd(&output[t * H + col], val);
}

// [B200 optimization — Phase 3b] Fused weighted scatter + bf16 cast.
//
// Old pipeline: zero fp32 output_f32 [T,H] → atomic-add fp32 → output.copy_(fp32→bf16).
// Three passes over ~400MB at seq=14107: zero-init, atomic-add, torch cast-copy.
//
// New pipeline: caller zeros the bf16 output once (output.zero_() = ~200MB memset),
// then we atomic-add bf16 pairs directly. One pass, fp32 output_f32 tensor deleted.
// Saves: 200MB allocation + 200MB torch cast-copy bandwidth.
//
// Native atomicAdd on __nv_bfloat162 (packed bf16x2) is hardware on SM90+, so no
// CAS loop. Precision loss per atomic is ~1 ULP in bf16. With topk=8 experts
// accumulating per token, total error is bounded by ~8 ULPs — tiny vs the contest
// tolerance (atol=1, rtol=0.3).
//
// Requires H % 2 == 0 (asserted; for DeepSeek H=7168 this holds).
__global__ void scatter_weighted_bf16_atomic_kernel(
    const float*        __restrict__ O,          // [N, H] fp32 — GEMM2 output
    const int*          __restrict__ token_ids,  // [N]  (T_max = padding sentinel)
    const float*        __restrict__ weights,    // [N]  (0.0 for padding)
    int N, int H, int T_max,
    __nv_bfloat16*      __restrict__ output)     // [T_max, H] bf16, zero-initialized
{
    // Each thread handles 2 adjacent bf16 columns (one packed bf16x2 atomic).
    int row  = blockIdx.z * gridDim.y + blockIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;  // in [0, H/2)
    int col  = col2 * 2;
    if (row >= N || col >= H) return;

    int t = token_ids[row];
    if (t < 0 || t >= T_max) return;
    float w = weights[row];
    if (w == 0.0f) return;

    float v0 = O[(size_t)row * H + col    ] * w;
    float v1 = O[(size_t)row * H + col + 1] * w;
    __nv_bfloat162 val = __floats2bfloat162_rn(v0, v1);

    auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(output + (size_t)t * H + col);
    atomicAdd(out_ptr, val);
}

// BF16-input variant: GEMM2 output is now bf16 (half the I/O at large seq).
__global__ void scatter_weighted_bf16_from_bf16_kernel(
    const __nv_bfloat16* __restrict__ O,          // [N, H] bf16 GEMM2 output
    const int*           __restrict__ token_ids,  // [N]
    const float*         __restrict__ weights,    // [N]
    int N, int H, int T_max,
    __nv_bfloat16*       __restrict__ output)     // [T_max, H] bf16
{
    int row  = blockIdx.z * gridDim.y + blockIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;
    int col  = col2 * 2;
    if (row >= N || col >= H) return;

    int t = token_ids[row];
    if (t < 0 || t >= T_max) return;
    float w = weights[row];
    if (w == 0.0f) return;

    // Vectorized bf16x2 load → fp32 multiply by weight → bf16x2 atomic add.
    auto* in_ptr = reinterpret_cast<const __nv_bfloat162*>(O + (size_t)row * H + col);
    __nv_bfloat162 v_in = *in_ptr;
    float v0 = __bfloat162float(v_in.x) * w;
    float v1 = __bfloat162float(v_in.y) * w;
    __nv_bfloat162 val = __floats2bfloat162_rn(v0, v1);

    auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(output + (size_t)t * H + col);
    atomicAdd(out_ptr, val);
}

// ═══════════════════════════════════════════════════════════════════════════════
// [B200 optimization — Phase #2] Device-side prefix scan over local-expert counts.
//
// Replaces the cudaStreamSynchronize + counts.cpu() + host prefix sum + 3 H2D
// memcpys at main.cpp lines 89-139 of v1. That sync alone cost ~20-50 µs which
// at seq=1 is >25% of flashinfer's total kernel latency.
//
// Layout: single block, single warp (32 threads = NUM_LOCAL_EXPERTS).
//   • Each thread loads its expert's count.
//   • Warp-level inclusive scan via __shfl_up_sync.
//   • Convert to exclusive offsets and write both unpadded + padded versions.
//   • Lane 31 writes the totals at index [E_local].
//
// Two unpadded copies are emitted because the downstream fill kernel mutates
// its copy via atomicAdd while build_padded_arrays reads the read-only copy.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void compute_offsets_kernel(
    const int* __restrict__ counts,
    int* __restrict__ unpadded_offsets,
    int* __restrict__ unpadded_offsets_atomic,
    int* __restrict__ padded_offsets,
    int pad_align)   // 4 (small seq) or 64 (large seq, M=64 GEMM2)
{
    static_assert(NUM_LOCAL_EXPERTS == 32, "kernel assumes E_local == 32 (single warp)");
    int tid = threadIdx.x;

    int c  = counts[tid];
    int pc = (c + pad_align - 1) & ~(pad_align - 1);

    // Inclusive warp scan via shfl_up_sync.
    int unp = c, pad = pc;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int u_n = __shfl_up_sync(0xffffffffu, unp, off);
        int p_n = __shfl_up_sync(0xffffffffu, pad, off);
        if (tid >= off) { unp += u_n; pad += p_n; }
    }
    // unp/pad now hold the inclusive prefix sum at this lane.
    // Exclusive = inclusive - my element.
    int unp_excl = unp - c;
    int pad_excl = pad - pc;

    unpadded_offsets[tid]        = unp_excl;
    unpadded_offsets_atomic[tid] = unp_excl;
    padded_offsets[tid]          = pad_excl;
    if (tid == 31) {
        unpadded_offsets[NUM_LOCAL_EXPERTS]        = unp;   // total_assign
        unpadded_offsets_atomic[NUM_LOCAL_EXPERTS] = unp;
        padded_offsets[NUM_LOCAL_EXPERTS]          = pad;   // padded_total
    }
}

void launch_compute_offsets(
    const int* counts,
    int* unpadded_offsets, int* unpadded_offsets_atomic, int* padded_offsets,
    cudaStream_t stream, int pad_align)
{
    compute_offsets_kernel<<<1, NUM_LOCAL_EXPERTS, 0, stream>>>(
        counts, unpadded_offsets, unpadded_offsets_atomic, padded_offsets, pad_align);
    CUDA_CHECK(cudaGetLastError());
}

// Launchers

void launch_noaux_routing_topk8(
    const float* routing_logits,
    const float* routing_bias,
    int T,
    float routed_scaling_factor,
    int* topk_idx,
    float* topk_w,
    cudaStream_t stream) {

  dim3 block(ROUTE_NUM_GROUP * 32); // 8 warps
  dim3 grid(T);
  noaux_routing_topk8_kernel<<<grid, block, 0, stream>>>(
      routing_logits, routing_bias, T, routed_scaling_factor, topk_idx, topk_w);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_hidden_block_scale(
    float* A_fp32,
    const float* hs_scale,
    int T,
    cudaStream_t stream) {
  int H = HIDDEN_SIZE;
  int64_t N64 = static_cast<int64_t>(T) * H;
  int threads = 256;
  int blocks = static_cast<int>((N64 + threads - 1) / threads);
  blocks = max(1, min(blocks, 65535));
  apply_hidden_block_scale_kernel<<<blocks, threads, 0, stream>>>(A_fp32, hs_scale, T, H);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_block_scale_128x128(
    float* M, int rows, int cols,
    const float* S, int S_rows, int S_cols,
    cudaStream_t stream) {

  dim3 grid(S_cols, S_rows);   // blocks in [cols/128, rows/128]
  dim3 block(32, 8);           // 256 threads
  apply_block_scale_128x128_kernel<<<grid, block, 0, stream>>>(M, rows, cols, S, S_rows, S_cols);
  CUDA_CHECK(cudaGetLastError());
}

void launch_count_local_assignments(
    const int* topk_idx, int T, int local_expert_offset,
    int* counts, cudaStream_t stream) {
  int threads = 256;
  int blocks = (T + threads - 1) / threads;
  count_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, T, local_expert_offset, counts);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fill_local_assignments(
    const int* topk_idx, const float* topk_w, int T, int local_expert_offset,
    int* offsets_inout, int* token_ids_out, float* token_w_out,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (T + threads - 1) / threads;
  fill_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, topk_w, T, local_expert_offset, offsets_inout, token_ids_out, token_w_out);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fill_local_assignments_with_inverse(
    const int* topk_idx, const float* topk_w, int T, int local_expert_offset,
    int* offsets_inout,
    const int* unpadded_offsets, const int* padded_offsets,
    int* token_ids_out, float* token_w_out,
    int* inv_padded_slot,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (T + threads - 1) / threads;
  fill_local_assignments_with_inverse_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, topk_w, T, local_expert_offset,
      offsets_inout, unpadded_offsets, padded_offsets,
      token_ids_out, token_w_out, inv_padded_slot);
  CUDA_CHECK(cudaGetLastError());
}

void launch_gather_rows(
    const float* A, const int* token_ids, int /*T*/, int Tk, int H,
    float* A_out, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid = make_row_grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    gather_rows_kernel<<<grid, block, 0, stream>>>(A, token_ids, 0, Tk, H, A_out);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_swiglu(
    const float* G1, int Tk, float* C, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid = make_row_grid((INTERMEDIATE_SIZE + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    swiglu_kernel<<<grid, block, 0, stream>>>(G1, Tk, C);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_accumulate_weighted_add(
    const float* O, const int* token_ids, const float* weights, int Tk, int H,
    float* output, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid = make_row_grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    accumulate_weighted_add_kernel<<<grid, block, 0, stream>>>(
        O, token_ids, weights, Tk, H, output);
    CUDA_CHECK(cudaGetLastError());
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Build padded arrays on GPU — removes the need for .cpu() calls.
// Produces padded_token_ids (T = padding sentinel), padded_safe_ids (0 for
// padding — safe for index_select), padded_valid (1.0 real / 0.0 padding),
// padded_token_wts (routing weight / 0.0 padding).
// ═══════════════════════════════════════════════════════════════════════════════

// Single kernel to initialize all 4 padded arrays with sentinels. Replaces
// 4 torch.full/zeros calls (saves ~45µs of launch overhead per forward).
// The build_padded_arrays_kernel then overwrites the [0, padded_total) range
// with real values; this init covers [padded_total, max_padded).
__global__ void init_padded_arrays_kernel(
    int max_padded, int T_sentinel,
    int*   __restrict__ padded_token_ids,
    int*   __restrict__ padded_safe_ids,
    float* __restrict__ padded_valid,
    float* __restrict__ padded_token_wts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_padded) return;
    padded_token_ids[i] = T_sentinel;
    padded_safe_ids[i] = 0;
    padded_valid[i] = 0.0f;
    padded_token_wts[i] = 0.0f;
}

void launch_init_padded_arrays(
    int max_padded, int T_sentinel,
    int* padded_token_ids, int* padded_safe_ids,
    float* padded_valid, float* padded_token_wts,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (max_padded + threads - 1) / threads;
    init_padded_arrays_kernel<<<blocks, threads, 0, stream>>>(
        max_padded, T_sentinel,
        padded_token_ids, padded_safe_ids, padded_valid, padded_token_wts);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void build_padded_arrays_kernel(
    const int*   __restrict__ unpadded_offsets, // [E+1] int32 cumulative
    const int*   __restrict__ padded_offsets,   // [E+1] int32 cumulative
    const int*   __restrict__ token_ids,         // [total_assign] int32
    const float* __restrict__ token_wts,         // [total_assign] float32
    int T_sentinel, int E_local,
    int*   padded_token_ids,
    int*   padded_safe_ids,
    float* padded_valid,
    float* padded_token_wts_out)
{
    // One block per expert; threads iterate over padded positions for that expert.
    int e = blockIdx.x;
    if (e >= E_local) return;

    int src_base = unpadded_offsets[e];
    int real_count = unpadded_offsets[e + 1] - src_base;
    int dst_base = padded_offsets[e];
    int padded_count = padded_offsets[e + 1] - dst_base;

    for (int j = threadIdx.x; j < padded_count; j += blockDim.x) {
        int dst = dst_base + j;
        if (j < real_count) {
            int tid = token_ids[src_base + j];
            float w = token_wts[src_base + j];
            padded_token_ids[dst]     = tid;
            padded_safe_ids[dst]      = tid;
            padded_valid[dst]         = 1.0f;
            padded_token_wts_out[dst] = w;
        } else {
            padded_token_ids[dst]     = T_sentinel;
            padded_safe_ids[dst]      = 0;
            padded_valid[dst]         = 0.0f;
            padded_token_wts_out[dst] = 0.0f;
        }
    }
}

void launch_build_padded_arrays(
    const int* unpadded_offsets, const int* padded_offsets,
    const int* token_ids, const float* token_wts,
    int T_sentinel, int E_local,
    int* padded_token_ids, int* padded_safe_ids,
    float* padded_valid, float* padded_token_wts_out,
    cudaStream_t stream)
{
    build_padded_arrays_kernel<<<E_local, 128, 0, stream>>>(
        unpadded_offsets, padded_offsets,
        token_ids, token_wts,
        T_sentinel, E_local,
        padded_token_ids, padded_safe_ids, padded_valid, padded_token_wts_out);
    CUDA_CHECK(cudaGetLastError());
}

// Atomic version — safe when multiple rows (different experts, same token)
// write to the same output token. Use T_max as padding sentinel.
void launch_accumulate_weighted_add_atomic(
    const float* O, const int* token_ids, const float* weights,
    int N, int H, int T_max, float* output, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid = make_row_grid((H + block.x - 1) / block.x, N);
  if (N > 0) {
    accumulate_weighted_add_atomic_kernel<<<grid, block, 0, stream>>>(
        O, token_ids, weights, N, H, T_max, output);
    CUDA_CHECK(cudaGetLastError());
  }
}

// [B200 optimization — Phase 3b] Launcher for the fused bf16-atomic scatter.
// Replaces launch_accumulate_weighted_add_atomic + output.copy_(fp32→bf16) pair.
void launch_scatter_weighted_bf16_atomic(
    const float* O, const int* token_ids, const float* weights,
    int N, int H, int T_max, __nv_bfloat16* output, cudaStream_t stream) {
    // H must be even (we atomic-add bf16 pairs). DeepSeek H=7168 → OK.
    // One thread handles 2 output columns, so grid-x dim is H/2 / block.x.
    dim3 block(256);
    dim3 grid = make_row_grid(((H / 2) + block.x - 1) / block.x, N);
    if (N > 0) {
        scatter_weighted_bf16_atomic_kernel<<<grid, block, 0, stream>>>(
            O, token_ids, weights, N, H, T_max, output);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_scatter_weighted_bf16_from_bf16(
    const __nv_bfloat16* O, const int* token_ids, const float* weights,
    int N, int H, int T_max, __nv_bfloat16* output, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid = make_row_grid(((H / 2) + block.x - 1) / block.x, N);
    if (N > 0) {
        scatter_weighted_bf16_from_bf16_kernel<<<grid, block, 0, stream>>>(
            O, token_ids, weights, N, H, T_max, output);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-atomic finalize: gather topK rows of GEMM2 output for each token, sum
// with weights, write bf16. Replaces atomic scatter (1.65ms → ~0.4ms at
// seq=14107). Mirrors flashinfer's finalizeKernelVecLoad approach.
//
// Grid: (H/(threads*2), T)  Block: 256 threads, each handles 2 cols (bf16x2).
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void finalize_weighted_bf16_kernel(
    const float*         __restrict__ gemm2_out,    // [N, H] fp32
    const int*           __restrict__ inv_slot,     // [T, topK]
    const float*         __restrict__ topk_weights, // [T, topK]
    int H,
    __nv_bfloat16*       __restrict__ output)       // [T, H]
{
    int t    = blockIdx.y;
    int col2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (col2 >= H) return;

    int   slots[ROUTE_TOP_K];
    float wts[ROUTE_TOP_K];
    #pragma unroll
    for (int k = 0; k < ROUTE_TOP_K; k++) {
        slots[k] = inv_slot[t * ROUTE_TOP_K + k];
        wts[k]   = topk_weights[t * ROUTE_TOP_K + k];
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (int k = 0; k < ROUTE_TOP_K; k++) {
        if (slots[k] < 0) continue;
        const float* row = gemm2_out + (size_t)slots[k] * H + col2;
        acc0 += wts[k] * row[0];
        acc1 += wts[k] * row[1];
    }

    auto* out_pair = reinterpret_cast<__nv_bfloat162*>(output + (size_t)t * H + col2);
    *out_pair = __floats2bfloat162_rn(acc0, acc1);
}

// BF16-input variant: GEMM2 with FI_ElementD=bfloat16_t outputs bf16, halving
// this kernel's read bandwidth (3.16GB → 1.58GB at seq=14107).
__global__ void finalize_weighted_bf16_from_bf16_kernel(
    const __nv_bfloat16* __restrict__ gemm2_out,    // [N, H] bf16
    const int*           __restrict__ inv_slot,     // [T, topK]
    const float*         __restrict__ topk_weights, // [T, topK]
    int H,
    __nv_bfloat16*       __restrict__ output)       // [T, H]
{
    int t    = blockIdx.y;
    int col2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (col2 >= H) return;

    int   slots[ROUTE_TOP_K];
    float wts[ROUTE_TOP_K];
    #pragma unroll
    for (int k = 0; k < ROUTE_TOP_K; k++) {
        slots[k] = inv_slot[t * ROUTE_TOP_K + k];
        wts[k]   = topk_weights[t * ROUTE_TOP_K + k];
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (int k = 0; k < ROUTE_TOP_K; k++) {
        if (slots[k] < 0) continue;
        // bf16x2 vectorized load (one LD.32 fetches 2 bf16 values).
        auto* row = reinterpret_cast<const __nv_bfloat162*>(
            gemm2_out + (size_t)slots[k] * H + col2);
        __nv_bfloat162 v = *row;
        acc0 += wts[k] * __bfloat162float(v.x);
        acc1 += wts[k] * __bfloat162float(v.y);
    }

    auto* out_pair = reinterpret_cast<__nv_bfloat162*>(output + (size_t)t * H + col2);
    *out_pair = __floats2bfloat162_rn(acc0, acc1);
}

void launch_finalize_weighted_bf16(
    const float* gemm2_out, const int* inv_slot, const float* topk_weights,
    int T, int H, __nv_bfloat16* output, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(((H / 2) + block.x - 1) / block.x, T);
    if (T > 0) {
        finalize_weighted_bf16_kernel<<<grid, block, 0, stream>>>(
            gemm2_out, inv_slot, topk_weights, H, output);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_finalize_weighted_bf16_from_bf16(
    const __nv_bfloat16* gemm2_out, const int* inv_slot, const float* topk_weights,
    int T, int H, __nv_bfloat16* output, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(((H / 2) + block.x - 1) / block.x, T);
    if (T > 0) {
        finalize_weighted_bf16_from_bf16_kernel<<<grid, block, 0, stream>>>(
            gemm2_out, inv_slot, topk_weights, H, output);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9) Fused FP8 dequant + block-scale for selected experts
// One kernel: reads FP8 byte, converts to float, multiplies by block-scale.
// Only processes active experts (those with tokens).
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float fp8_e4m3fn_to_float_fast(uint8_t x) {
    unsigned sign = x >> 7;
    unsigned exp  = (x >> 3) & 0x0Fu;
    unsigned mant = x & 0x07u;
    if (exp == 0x0Fu && mant == 0x07u) return 0.0f; // NaN → 0
    float val;
    if (exp == 0u) val = ldexpf((float)mant, -9);
    else val = ldexpf((float)(mant | 8u), (int)exp - 10);
    return sign ? -val : val;
}

// FP32 version
__global__ void fused_dequant_experts_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int* __restrict__ active_eids,
    int rows, int cols, int scale_cols_per_block)
{
    int active_idx = blockIdx.z;
    int e = active_eids[active_idx];
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    int64_t expert_stride = (int64_t)rows * cols;
    int rb = r / 128;
    int cb = c / 128;
    int scale_rows = rows / 128;

    float s = scale[(int64_t)e * scale_rows * scale_cols_per_block + rb * scale_cols_per_block + cb];
    uint8_t fp8_val = w_fp8[(int64_t)e * expert_stride + r * cols + c];
    out[(int64_t)e * expert_stride + r * cols + c] = fp8_e4m3fn_to_float_fast(fp8_val) * s;
}

void launch_fused_dequant_experts(
    const uint8_t* w_fp8, const float* scale, float* out,
    const int* active_eids, int num_active, int rows, int cols,
    cudaStream_t stream) {
    if (num_active == 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows, num_active);
    fused_dequant_experts_kernel<<<grid, block, 0, stream>>>(
        w_fp8, scale, out, active_eids, rows, cols, cols / 128);
    CUDA_CHECK(cudaGetLastError());
}

// BF16 version — halves output bandwidth
__global__ void fused_dequant_experts_bf16_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ active_eids,
    int rows, int cols, int scale_cols_per_block)
{
    int active_idx = blockIdx.z;
    int e = active_eids[active_idx];
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    int64_t expert_stride = (int64_t)rows * cols;
    int rb = r / 128, cb = c / 128;
    int scale_rows = rows / 128;

    float s = scale[(int64_t)e * scale_rows * scale_cols_per_block + rb * scale_cols_per_block + cb];
    uint8_t fp8_val = w_fp8[(int64_t)e * expert_stride + r * cols + c];
    out[(int64_t)e * expert_stride + r * cols + c] = __float2bfloat16(fp8_e4m3fn_to_float_fast(fp8_val) * s);
}

void launch_fused_dequant_experts_bf16(
    const uint8_t* w_fp8, const float* scale, void* out,
    const int* active_eids, int num_active, int rows, int cols,
    cudaStream_t stream) {
    if (num_active == 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows, num_active);
    fused_dequant_experts_bf16_kernel<<<grid, block, 0, stream>>>(
        w_fp8, scale, (__nv_bfloat16*)out, active_eids, rows, cols, cols / 128);
    CUDA_CHECK(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10) FP8 blockwise quantization
// Each block handles one 128-element segment of one row.
// Computes absmax per block, derives scale, quantizes to FP8 e4m3fn.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float x) {
    x = fminf(fmaxf(x, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__global__ void quantize_fp8_blockwise_kernel(
    const float* __restrict__ input,     // [rows, cols]
    uint8_t* __restrict__ output_fp8,    // [rows, cols]
    float* __restrict__ scales,          // [cols/128, rows] COLUMN-MAJOR
    int rows, int cols)
{
    int row = blockIdx.z * gridDim.y + blockIdx.y;
    int block_idx = blockIdx.x;  // which 128-element block within the row
    int tid = threadIdx.x;

    if (row >= rows) return;
    int col_start = block_idx * 128;
    if (col_start >= cols) return;

    // Compute absmax for this 128-element block
    float local_max = 0.0f;
    for (int i = tid; i < 128 && col_start + i < cols; i += blockDim.x) {
        float v = fabsf(input[row * cols + col_start + i]);
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce
    local_max = warp_max(local_max);

    // Block reduce (assume blockDim.x <= 128, single warp is enough)
    __shared__ float s_max;
    if (tid == 0) s_max = 0.0f;
    __syncthreads();
    if (tid % 32 == 0) atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    float absmax = s_max;

    // Scale = absmax / FP8_MAX (avoid division by zero)
    float scale = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale in COLUMN-MAJOR [cols/128, rows]: index = block_idx * rows + row
    if (tid == 0) {
        scales[block_idx * rows + row] = scale;
    }

    // Quantize
    for (int i = tid; i < 128 && col_start + i < cols; i += blockDim.x) {
        float v = input[row * cols + col_start + i] * inv_scale;
        output_fp8[row * cols + col_start + i] = float_to_fp8_e4m3(v);
    }
}

void launch_quantize_fp8_blockwise(
    const float* input, uint8_t* output_fp8, float* scales,
    int rows, int cols, cudaStream_t stream) {
    int num_blocks = (cols + 127) / 128;
    dim3 block(128);
    dim3 grid = make_row_grid(num_blocks, rows);
    quantize_fp8_blockwise_kernel<<<grid, block, 0, stream>>>(
        input, output_fp8, scales, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9c) Fused SwiGLU + FP8 blockwise quantization — replaces launch_swiglu for the
//     FP8 GEMM2 path.
//
// Why fuse: Without this, we'd have to (1) write fp32 SwiGLU output to HBM,
// (2) read it back in a separate quantize pass, (3) write fp8+scales. One kernel
// keeps everything in registers / smem: read G1 → compute silu*gate → block-reduce
// absmax → write fp8+scale. Saves one 2*I*padded*4B HBM round-trip.
//
// Why per-128-block absmax: matches CUTLASS Sm100BlockwiseScaleConfig<1,128,128>
// granularity exactly. Each of the 128 threads in the block computes ONE output
// element, then they reduce absmax among themselves. The 128-elt reduction gives
// us one fp8 scale per (row, 128-col-block), which is M=1 per-token granularity
// on the GEMM2 K-dim — same recipe DeepSeek uses for hidden_states_scale.
//
// Scale layout matches GEMM1's SFA: [cols/128, rows] column-major. Feeds directly
// into cutlass_moe_gemm_fp8_blockwise's SFA argument.
//
// Contest tolerance is atol=1 rtol=0.3 matched@0.9 — per-128-block absmax passes
// with huge margin even at seq=1 where outliers could skew the per-block scale.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void swiglu_quantize_fp8_kernel(
    const float* __restrict__ G1,      // [padded_total, 2*I] fp32, GEMM1 output
    uint8_t*    __restrict__ C_fp8,       // [padded_total, I] fp8 e4m3
    float*      __restrict__ scales,      // [I/128, padded_total] fp32 column-major
    int padded_total)
{
    const int I = INTERMEDIATE_SIZE;                   // 2048
    // [bugfix] padded_total can exceed CUDA's gridDim.y max (65535) at seq=11948+,
    // so we split the row dim across (y, z). Match the make_row_grid convention
    // used elsewhere in this file: row = blockIdx.z * gridDim.y + blockIdx.y.
    const int row       = blockIdx.z * gridDim.y + blockIdx.y;
    const int block_col = blockIdx.x;                  // 0 .. I/128-1
    const int tid       = threadIdx.x;                 // 0 .. 127
    if (row >= padded_total) return;

    const int col = block_col * 128 + tid;
    const float* g1_row = G1 + (size_t)row * (2 * I);

    // SwiGLU: out = silu(x2) * x1, where x1=gate (first I), x2=up (second I).
    float x1 = g1_row[col];
    float x2 = g1_row[col + I];
    float silu = x2 / (1.0f + __expf(-x2));
    float out  = silu * x1;

    // Block-reduce absmax across the 128 threads that share one output block.
    // Warp reduce first (32 lanes via shfl), then cross-warp via smem (4 warps).
    float m = fabsf(out);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    }
    __shared__ float warp_max_shared[4];
    if ((tid & 31) == 0) warp_max_shared[tid >> 5] = m;
    __syncthreads();

    // Final reduce: warp 0 merges the 4 warp-level maxes.
    if ((tid >> 5) == 0) {
        float v = (tid < 4) ? warp_max_shared[tid] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (tid == 0) warp_max_shared[0] = v;
    }
    __syncthreads();
    const float absmax = warp_max_shared[0];

    // Scale = absmax / 448 (FP8_E4M3_MAX). Guard against all-zero blocks so the
    // GEMM doesn't see scale=0 (would zero the accumulator on that block).
    const float scale     = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
    const float inv_scale = 1.0f / scale;

    C_fp8[(size_t)row * I + col] = float_to_fp8_e4m3(out * inv_scale);

    // One scale per 128-col-block, one thread writes. Column-major layout:
    // scales[block_col, row] lives at block_col * padded_total + row.
    if (tid == 0) {
        scales[(size_t)block_col * padded_total + row] = scale;
    }
}

void launch_swiglu_quantize_fp8(
    const float* G1, int padded_total,
    uint8_t* C_fp8, float* scales, cudaStream_t stream)
{
    // Grid: one block per (row, 128-col-output-block). Block dim = 128 so exactly
    // one thread per output element in the block. Keeps absmax reduction local.
    //
    // [B200 optimization — Phase 1b] PDL: this kernel's output feeds GEMM2, a
    // PDL-aware CUTLASS kernel. Emitting the PDL "trigger" here lets GEMM2's
    // pointer-setup + TMA-descriptor init overlap with our tail writes.
    //
    // [bugfix] At seq=11948+ padded_total exceeds gridDim.y max (65535), so we
    // route through make_row_grid which splits the row dim across (y, z).
    dim3 grid = make_row_grid(INTERMEDIATE_SIZE / 128, padded_total);
    dim3 block(128);
    LAUNCH_PDL(swiglu_quantize_fp8_kernel, grid, block, 0, stream,
               G1, C_fp8, scales, padded_total);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9e) In-place FP8 dequant + M=64 requant for GEMM1 activations.
//
// The sorted activations are FP8 with per-token per-K-block scales in
// sfa[K//128, padded_total] (col-major). This kernel "absorbs" those per-token
// scales into the FP8 values and produces coarser M=64 block scales, eliminating
// 3,584 strided scale loads per GEMM1 tile (down to 56).
//
// Grid: (H/128, padded_total/64)  Block: 128 threads (one per K-col in block).
// padded_total must be multiple of 64 (guaranteed by large-seq padding).
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void fp8_absorb_token_scale_m64_kernel(
    uint8_t*     __restrict__ act_fp8,      // [padded_total, H] fp8 e4m3, in-place
    const float* __restrict__ per_tok_sfa,  // [H//128, padded_total] col-major
    float*       __restrict__ new_sfa_m64,  // [H//128, padded_total//64] output
    int padded_total)
{
    const int H        = HIDDEN_SIZE;
    const int m_tile   = blockIdx.y;    // 0 .. padded_total/64 - 1
    const int k_block  = blockIdx.x;    // 0 .. H/128 - 1
    const int tid      = threadIdx.x;   // 0 .. 127 (one per K-col in block)
    const int base_row = m_tile * 64;
    const int col      = k_block * 128 + tid;

    // Per-token SFA for this K-block, 64 rows: col-major → stride 1 in M.
    const float* sfa_row = per_tok_sfa + (size_t)k_block * padded_total + base_row;

    // Load 64 fp8 values and dequant.
    float vals[64];
    float my_max = 0.0f;
    #pragma unroll 8
    for (int r = 0; r < 64; r++) {
        float fp8v = fp8_e4m3fn_to_float_fast(act_fp8[(size_t)(base_row + r) * H + col]);
        float v    = fp8v * sfa_row[r];  // per-token dequant
        vals[r]    = v;
        my_max     = fmaxf(my_max, fabsf(v));
    }

    // 128-thread warp reduce.
    float m = my_max;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    __shared__ float warp_max[4];
    if ((tid & 31) == 0) warp_max[tid >> 5] = m;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < 4) ? warp_max[tid] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (tid == 0) warp_max[0] = v;
    }
    __syncthreads();
    const float absmax    = warp_max[0];
    const float new_scale = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
    const float inv_scale = 1.0f / new_scale;

    // Re-quantize FP8 in-place.
    #pragma unroll 8
    for (int r = 0; r < 64; r++)
        act_fp8[(size_t)(base_row + r) * H + col] = float_to_fp8_e4m3(vals[r] * inv_scale);

    // Write M=64 block scale: [H//128, padded_total//64] col-major.
    if (tid == 0)
        new_sfa_m64[(size_t)k_block * (padded_total / 64) + m_tile] = new_scale;
}

void launch_fp8_absorb_token_scale_m64(
    uint8_t* act_fp8, const float* per_tok_sfa,
    float* new_sfa_m64, int padded_total, cudaStream_t stream)
{
    dim3 grid(HIDDEN_SIZE / 128, padded_total / 64);
    dim3 block(128);
    fp8_absorb_token_scale_m64_kernel<<<grid, block, 0, stream>>>(
        act_fp8, per_tok_sfa, new_sfa_m64, padded_total);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9d) Fused SwiGLU + FP8 quantize with M=64 tile-granularity A scales.
//
// Replaces 9c for GEMM2 when fi_gemm_m64 (Sm100BlockwiseScaleConfig<64,...>)
// is used. One CUDA block handles 64 rows × 128 K-cols, emitting a single
// scale for the whole tile instead of one per token row.
//
// Why M=64 reduces register pressure: CUTLASS mainloop loads 1 scale per
// K-block (instead of tile_M=64 scales), eliminating 63 per-K-block loads and
// the bookkeeping registers that track them. Measured: 168→? reg/thread.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void swiglu_quantize_fp8_m64_kernel(
    const float* __restrict__ G1,      // [padded_total, 2*I] fp32
    uint8_t*    __restrict__ C_fp8,       // [padded_total, I]
    float*      __restrict__ scales,      // [I/128, padded_total/64] col-major
    int padded_total)
{
    const int I         = INTERMEDIATE_SIZE;
    const int row_tile  = blockIdx.y;   // 0 .. padded_total/64 - 1
    const int block_col = blockIdx.x;   // 0 .. I/128 - 1
    const int tid       = threadIdx.x;  // 0 .. 127 (one per col in the 128-wide K-block)
    const int base_row  = row_tile * 64;
    const int col       = block_col * 128 + tid;

    // Load and compute SwiGLU for all 64 rows in this column.
    // Keep results in registers for the second write pass.
    float vals[64];
    float my_max = 0.0f;
    #pragma unroll 8
    for (int r = 0; r < 64; r++) {
        const float* row_ptr = G1 + (size_t)(base_row + r) * (2 * I);
        float x1   = row_ptr[col];
        float x2   = row_ptr[col + I];
        float silu = x2 / (1.0f + __expf(-x2));
        float out  = silu * x1;
        vals[r]    = out;
        my_max     = fmaxf(my_max, fabsf(out));
    }

    // Warp-reduce my_max (128 threads = 4 warps).
    float m = my_max;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    __shared__ float warp_max[4];
    if ((tid & 31) == 0) warp_max[tid >> 5] = m;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < 4) ? warp_max[tid] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (tid == 0) warp_max[0] = v;
    }
    __syncthreads();
    const float absmax   = warp_max[0];
    const float scale    = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
    const float inv_scale = 1.0f / scale;

    // Write 64 FP8 values for my column.
    #pragma unroll 8
    for (int r = 0; r < 64; r++)
        C_fp8[(size_t)(base_row + r) * I + col] = float_to_fp8_e4m3(vals[r] * inv_scale);

    // One scale per 64×128 tile. Layout: [I/128, padded_total/64] col-major.
    // scales[block_col, row_tile] → index = block_col * (padded_total/64) + row_tile
    if (tid == 0)
        scales[(size_t)block_col * (padded_total / 64) + row_tile] = scale;
}

void launch_swiglu_quantize_fp8_m64(
    const float* G1, int padded_total,
    uint8_t* C_fp8, float* scales, cudaStream_t stream)
{
    // padded_total must be a multiple of 64 (guaranteed by compute_offsets_kernel).
    // PDL: the consumer GEMM2 (CUTLASS PDL-aware) starts its prologue before
    // this kernel's tail flushes, overlapping ~0.1-0.3ms with GEMM2 setup.
    dim3 grid(INTERMEDIATE_SIZE / 128, padded_total / 64);
    dim3 block(128);
    LAUNCH_PDL(swiglu_quantize_fp8_m64_kernel, grid, block, 0, stream,
               G1, C_fp8, scales, padded_total);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10) FP8 byte-level gather — vectorized with uint4 (16 bytes per thread).
//
// [B200 optimization — Phase #3 partial] The pure win (in-kernel gather inside
// CUTLASS GEMM1 mainloop) requires writing a custom CollectiveMainloop with
// TMA-with-indirection — multi-day work that can't be done blindly without a
// GPU to test. This is the pragmatic alternative: keep the gather as a
// separate kernel but issue 16-byte aligned vectorized loads. Reduces memory
// transactions ~16×, aligns with HBM3e's preferred 32B/64B sectors, and lets
// the compiler emit ldgsts (async copy) on B200.
//
// Requires K % 16 == 0 (DeepSeek H=7168 → 16 OK; intermediate 2048 → 16 OK).
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void gather_fp8_rows_vec_kernel(
    const uint8_t* __restrict__ src,
    const int*     __restrict__ ids,
    int T, int Tk, int K,
    uint8_t*       __restrict__ dst)
{
    int row    = blockIdx.z * gridDim.y + blockIdx.y;
    int vec_col = blockIdx.x * blockDim.x + threadIdx.x;  // each thread = 16 bytes
    int K_vec   = K / 16;
    if (row >= Tk || vec_col >= K_vec) return;

    int src_row = ids[row];
    bool valid  = (src_row >= 0 && src_row < T);

    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint4*       dst_v = reinterpret_cast<uint4*>(dst);

    uint4 v;
    if (valid) {
        v = src_v[(size_t)src_row * K_vec + vec_col];
    } else {
        v = make_uint4(0, 0, 0, 0);
    }
    dst_v[(size_t)row * K_vec + vec_col] = v;
}

void launch_gather_fp8_rows(
    const uint8_t* src, const int* token_ids,
    int T, int Tk, int K, uint8_t* dst, cudaStream_t stream) {
    // [B200 optimization — Phase 1b] PDL: output feeds GEMM1 directly. GEMM1's
    // launch_with_pdl=true lets it start prologue while our gather tail flushes.
    //
    // Grid X is K_vec / block = (K/16) / 256. For H=7168, K_vec=448 → 2 blocks.
    if (Tk <= 0) return;
    int K_vec = K / 16;
    dim3 block(256);
    dim3 grid = make_row_grid((K_vec + 255) / 256, Tk);
    LAUNCH_PDL(gather_fp8_rows_vec_kernel, grid, block, 0, stream,
               src, token_ids, T, Tk, K, dst);
}
