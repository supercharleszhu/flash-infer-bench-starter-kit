#ifndef MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_
#define MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_

#include <cuda_runtime.h>
#include <cstdint>

// B200-tuned constants for this specialized kernel
static constexpr int HIDDEN_SIZE = 7168;        // H
static constexpr int INTERMEDIATE_SIZE = 2048;  // I
static constexpr int GEMM1_OUT_SIZE = 4096;     // 2 * I
static constexpr int NUM_EXPERTS_GLOBAL = 256;  // E_global
static constexpr int NUM_LOCAL_EXPERTS = 32;    // E_local
static constexpr int BLOCK_SIZE_128 = 128;

static constexpr int NUM_HIDDEN_BLOCKS = 56;         // H / 128
static constexpr int NUM_INTERMEDIATE_BLOCKS = 16;   // I / 128
static constexpr int NUM_GEMM1_OUT_BLOCKS = 32;      // (2*I)/128

// DeepSeek routing constants
static constexpr int ROUTE_TOP_K = 8;
static constexpr int ROUTE_NUM_GROUP = 8;
static constexpr int ROUTE_GROUP_SIZE = 32;    // NUM_EXPERTS_GLOBAL / ROUTE_NUM_GROUP
static constexpr int ROUTE_TOPK_GROUP = 4;

// Error check macro
#define CUDA_CHECK(status) \
  do { \
    cudaError_t err__ = (status); \
    if (err__ != cudaSuccess) { \
      fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    } \
  } while (0)

// Kernel launchers

// 1) No-aux routing with group-top2 and global top-k=8
void launch_noaux_routing_topk8(
    const float* routing_logits,   // [T, 256]
    const float* routing_bias,     // [256] (float32)
    int T,                         // seq_len
    float routed_scaling_factor,
    int* __restrict__ topk_idx,    // [T, 8] (int32)
    float* __restrict__ topk_w,    // [T, 8] (float32)
    cudaStream_t stream);

// 2) Hidden states block-scale application (after FP8 -> float32 conversion)
void launch_apply_hidden_block_scale(
    float* __restrict__ A_fp32,     // [T, H], in-place
    const float* __restrict__ hs_scale, // [H/128, T] contiguous
    int T,
    cudaStream_t stream);

// 3) Apply 128x128 block scale to 2D matrix (in-place)
void launch_apply_block_scale_128x128(
    float* __restrict__ M,          // [rows, cols], row-major
    int rows,                       // multiple of 128
    int cols,                       // multiple of 128
    const float* __restrict__ S,    // [rows/128, cols/128], row-major
    int S_rows,                     // rows/128
    int S_cols,                     // cols/128
    cudaStream_t stream);

// 4) Count assignments per local expert
void launch_count_local_assignments(
    const int* __restrict__ topk_idx,  // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ counts,          // [32], zero-initialized
    cudaStream_t stream);

// 5) Fill flat assignment lists using prefix offsets (atomic on-device)
void launch_fill_local_assignments(
    const int* __restrict__ topk_idx,   // [T, 8]
    const float* __restrict__ topk_w,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ offsets_inout,    // [32], device-side running offsets (initialized with prefix "offsets")
    int* __restrict__ token_ids_out,    // [total_assignments]
    float* __restrict__ token_w_out,    // [total_assignments]
    cudaStream_t stream);

// 6) Gather rows from [T, H] by token_ids to a compact [Tk, H]
void launch_gather_rows(
    const float* __restrict__ A,      // [T, H]
    const int* __restrict__ token_ids,// [Tk]
    int T, int Tk, int H,
    float* __restrict__ A_out,        // [Tk, H]
    cudaStream_t stream);

// 7) SwiGLU on GEMM1 output: C = silu(G1[:, I:]) * G1[:, :I]
void launch_swiglu(
    const float* __restrict__ G1,   // [Tk, 4096]
    int Tk,
    float* __restrict__ C,          // [Tk, 2048]
    cudaStream_t stream);

// 8) Accumulate O[Tk,H] into output[T,H] by token_ids and weights (no atomics if sequential per expert)
void launch_accumulate_weighted_add(
    const float* __restrict__ O,        // [Tk, H]
    const int* __restrict__ token_ids,  // [Tk]
    const float* __restrict__ weights,  // [Tk]
    int Tk, int H,
    float* __restrict__ output,         // [T, H]
    cudaStream_t stream);

// 8c) Build padded arrays on GPU (eliminates .cpu() syncs in main.cpp).
// Inputs are GPU arrays; outputs are GPU arrays of size padded_offsets[E_local].
void launch_build_padded_arrays(
    const int* unpadded_offsets,      // [E+1] int32 cumulative
    const int* padded_offsets,        // [E+1] int32 cumulative
    const int* token_ids,             // [total_assign] int32
    const float* token_wts,           // [total_assign] float32
    int T_sentinel, int E_local,
    int* padded_token_ids,            // out: [padded_total] T for padding
    int* padded_safe_ids,             // out: [padded_total] 0 for padding
    float* padded_valid,              // out: [padded_total] 0.0 for padding
    float* padded_token_wts_out,      // out: [padded_total] 0.0 for padding
    cudaStream_t stream);

// 8b) Atomic version — safe across experts sharing tokens. Row i is skipped if
// token_ids[i] >= T_max or out of range, and if weights[i] == 0.
void launch_accumulate_weighted_add_atomic(
    const float* __restrict__ O,        // [N, H] (N = padded_total)
    const int* __restrict__ token_ids,  // [N]   (T_max = padding sentinel)
    const float* __restrict__ weights,  // [N]   (0.0 for padding)
    int N, int H, int T_max,
    float* __restrict__ output,         // [T_max, H]
    cudaStream_t stream);

// 9a) Fused FP8 dequant to BF16 (half bandwidth vs FP32)
void launch_fused_dequant_experts_bf16(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ scale,
    void* __restrict__ out_bf16,
    const int* __restrict__ active_eids,
    int num_active, int rows, int cols,
    cudaStream_t stream);

// 9b) Fused FP8 dequant + block-scale for selected experts (FP32 version)
void launch_fused_dequant_experts(
    const uint8_t* __restrict__ w_fp8,   // [E, rows, cols] raw fp8
    const float* __restrict__ scale,      // [E, rows/128, cols/128] float32
    float* __restrict__ out,              // [E, rows, cols] float32
    const int* __restrict__ active_eids,  // [num_active] expert indices
    int num_active, int rows, int cols,
    cudaStream_t stream);

// 10) FP8 blockwise quantization: FP32 → FP8 e4m3fn + per-128-block scales
void launch_quantize_fp8_blockwise(
    const float* __restrict__ input,    // [rows, cols]
    uint8_t* __restrict__ output_fp8,   // [rows, cols] raw fp8 bytes
    float* __restrict__ scales,         // [cols/128, rows] COLUMN-MAJOR (CUTLASS SFA layout)
    int rows, int cols,
    cudaStream_t stream);

// 10) Gather FP8 bytes directly (no FP32 roundtrip)
void launch_gather_fp8_rows(
    const uint8_t* __restrict__ src,    // [T, K] raw fp8 bytes
    const int* __restrict__ token_ids,  // [Tk]
    int T, int Tk, int K,
    uint8_t* __restrict__ dst,          // [Tk, K]
    cudaStream_t stream);

#endif // MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_
