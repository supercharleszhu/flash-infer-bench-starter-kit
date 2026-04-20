#include "kernel.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

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

  // Merge 64 candidates to top-8 globally
  if (threadIdx.x == 0) {
    float temp_val[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    int   temp_idx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    float temp_snb[ROUTE_NUM_GROUP * ROUTE_TOP_K];

    #pragma unroll
    for (int i = 0; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i) {
      temp_val[i] = warpCandVal[i];
      temp_idx[i] = warpCandIdx[i];
      temp_snb[i] = warpCandSNoBias[i];
    }

    float sel_s[ROUTE_TOP_K];
    int sel_idx[ROUTE_TOP_K];

    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      int best_i = 0;
      float best_v = temp_val[0];
      #pragma unroll
      for (int i = 1; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i) {
        if (temp_val[i] > best_v) { best_v = temp_val[i]; best_i = i; }
      }
      sel_idx[j] = temp_idx[best_i];
      sel_s[j] = temp_snb[best_i];
      temp_val[best_i] = -CUDART_INF_F;
    }

    // Normalize weights using s (no bias)
    float sumw = 0.f;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) sumw += sel_s[j];
    sumw = fmaxf(sumw, 1e-20f);
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      float w = (sel_s[j] / sumw) * routed_scaling_factor;
      topk_idx[t * ROUTE_TOP_K + j] = sel_idx[j];
      topk_w[t * ROUTE_TOP_K + j] = w;
    }
  }
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
// 10) FP8 byte-level gather
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void gather_fp8_rows_kernel(
    const uint8_t* __restrict__ src, const int* __restrict__ ids,
    int T, int Tk, int K, uint8_t* __restrict__ dst) {
    int row = blockIdx.z * gridDim.y + blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Tk || col >= K) return;
    int src_row = ids[row];
    dst[row * K + col] = (src_row >= 0 && src_row < T) ? src[src_row * K + col] : 0;
}

void launch_gather_fp8_rows(
    const uint8_t* src, const int* token_ids,
    int T, int Tk, int K, uint8_t* dst, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid = make_row_grid((K + 255) / 256, Tk);
    if (Tk > 0) {
        gather_fp8_rows_kernel<<<grid, block, 0, stream>>>(src, token_ids, T, Tk, K, dst);
        CUDA_CHECK(cudaGetLastError());
    }
}
