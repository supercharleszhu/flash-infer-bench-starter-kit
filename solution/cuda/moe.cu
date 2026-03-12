/*
 * CUDA C++ MoE kernel — FP8 block-scale DeepSeek-V3 routing, torch binding.
 *
 * Definition : moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 * Language   : cuda  (TorchBuilder)
 * Binding    : torch
 * DPS        : true  (output pre-allocated bfloat16 [T, H], passed as last arg)
 * Entry      : moe.cu::kernel
 *
 * Key optimizations over the Python reference:
 *   1. Fused FP8 dequant kernels — single pass, no repeat_interleave temporaries.
 *   2. C++ routing — eliminates Python interpreter overhead.
 *   3. C++ expert loop — no Python overhead, uses ATen matmul (cuBLAS).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// ---------------------------------------------------------------------------
// FP8 E4M3FN conversion (manual — no cuda_fp8.h dependency)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t x) {
    // NV FP8 E4M3FN: sign=1, exp=4 (bias 7), mantissa=3; NaN=0x7F/0xFF, no Inf.
    uint32_t sign = x >> 7;
    uint32_t exp  = (x >> 3) & 0x0Fu;
    uint32_t mant = x & 0x07u;
    if (exp == 0x0Fu && mant == 0x07u) {
        // NaN
        return __int_as_float(0x7FC00000u);
    }
    float val;
    if (exp == 0u) {
        // Subnormal: (-1)^s * 2^(-6) * (mant/8)  =  mant * 2^(-9)
        val = ldexpf((float)mant, -9);
    } else {
        // Normal: (-1)^s * 2^(exp-7) * (1 + mant/8)  =  (mant+8) * 2^(exp-10)
        val = ldexpf((float)(mant | 8u), (int)exp - 10);
    }
    return sign ? -val : val;
}

// ---------------------------------------------------------------------------
// Kernel 1 — fused FP8 dequant for hidden_states
//   hs_fp8   : [T, H]        stored as uint8 (fp8_e4m3fn)
//   scale    : [H/BLOCK, T]  float32
//   out      : [T, H]        float32
// Each thread handles one (t, h) element.
// ---------------------------------------------------------------------------

__global__ void fp8_dequant_hs_kernel(
    const uint8_t* __restrict__ hs_fp8,
    const float*   __restrict__ scale,
    float*         __restrict__ out,
    int T, int H, int BLOCK)
{
    int t = blockIdx.x;
    int h = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || h >= H) return;
    int block_idx = h / BLOCK;
    float s = scale[block_idx * T + t];   // layout: [H/BLOCK, T]
    out[t * H + h] = fp8_e4m3fn_to_float(hs_fp8[t * H + h]) * s;
}

// ---------------------------------------------------------------------------
// Kernel 2 — fused FP8 dequant for weight tensors
//   w_fp8  : [E, rows, cols]           stored as uint8
//   scale  : [E, rows/BLOCK, cols/BLOCK] float32
//   out    : [E, rows, cols]           float32
// ---------------------------------------------------------------------------

__global__ void fp8_dequant_weight_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float*   __restrict__ scale,
    float*         __restrict__ out,
    int rows, int cols, int BLOCK,  // per-expert dims; E handled by blockIdx.z
    int scale_rows, int scale_cols)
{
    int e = blockIdx.z;
    int r = blockIdx.y;
    int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    int64_t expert_stride = (int64_t)rows * cols;
    int rb = r / BLOCK;
    int cb = c / BLOCK;
    float s = scale[(int64_t)e * scale_rows * scale_cols + rb * scale_cols + cb];
    out[(int64_t)e * expert_stride + r * cols + c] =
        fp8_e4m3fn_to_float(w_fp8[(int64_t)e * expert_stride + r * cols + c]) * s;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static at::Tensor dequant_hs(
    const at::Tensor& hs,    // [T, H]  fp8_e4m3fn → uint8 view
    const at::Tensor& scale, // [H/BLOCK, T] float32
    int T, int H, int BLOCK,
    cudaStream_t stream)
{
    auto out = at::empty({T, H}, hs.options().dtype(at::kFloat));
    dim3 block(256);
    dim3 grid(T, (H + 255) / 256);
    fp8_dequant_hs_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(hs.data_ptr()),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        T, H, BLOCK);
    return out;
}

static at::Tensor dequant_weight(
    const at::Tensor& w,     // [E, rows, cols] fp8_e4m3fn
    const at::Tensor& scale, // [E, rows/BLOCK, cols/BLOCK] float32
    int E, int rows, int cols, int BLOCK,
    cudaStream_t stream)
{
    auto out = at::empty({E, rows, cols}, w.options().dtype(at::kFloat));
    int scale_rows = rows / BLOCK;
    int scale_cols = cols / BLOCK;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows, E);
    fp8_dequant_weight_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(w.data_ptr()),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        rows, cols, BLOCK, scale_rows, scale_cols);
    return out;
}

// ---------------------------------------------------------------------------
// Main kernel entry (DPS=true: output is last argument)
// ---------------------------------------------------------------------------

void kernel(
    at::Tensor routing_logits,        // [T, E_global]  float32
    at::Tensor routing_bias,          // [E_global]      bfloat16
    at::Tensor hidden_states,         // [T, H]          fp8_e4m3fn
    at::Tensor hidden_states_scale,   // [H/128, T]      float32
    at::Tensor gemm1_weights,         // [E_local, 2I, H] fp8_e4m3fn
    at::Tensor gemm1_weights_scale,   // [E_local, 2I/128, H/128] float32
    at::Tensor gemm2_weights,         // [E_local, H, I]  fp8_e4m3fn
    at::Tensor gemm2_weights_scale,   // [E_local, H/128, I/128]  float32
    int64_t    local_expert_offset,
    double     routed_scaling_factor,
    at::Tensor output                 // [T, H]  bfloat16  (DPS)
) {
    constexpr int H       = 7168;
    constexpr int I       = 2048;
    constexpr int BLOCK   = 128;
    constexpr int E_GLOBAL = 256;
    constexpr int TOP_K   = 8;
    constexpr int N_GROUP = 8;
    constexpr int TOPK_GROUP = 4;

    const int T       = (int)routing_logits.size(0);
    const int E_local = (int)gemm1_weights.size(0);
    const int local_start = (int)local_expert_offset;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ── 1. FP8 dequantization (fused kernels, no repeat_interleave temps) ──────

    // A: [T, H] float32
    at::Tensor A = dequant_hs(hidden_states, hidden_states_scale.to(at::kFloat),
                               T, H, BLOCK, stream);

    // W13: [E_local, 2I, H] float32
    at::Tensor W13 = dequant_weight(gemm1_weights, gemm1_weights_scale.to(at::kFloat),
                                    E_local, 2 * I, H, BLOCK, stream);

    // W2: [E_local, H, I] float32
    at::Tensor W2 = dequant_weight(gemm2_weights, gemm2_weights_scale.to(at::kFloat),
                                   E_local, H, I, BLOCK, stream);

    // ── 2. No-aux routing (C++ ATen, no Python overhead) ─────────────────────

    at::Tensor logits = routing_logits.to(at::kFloat);       // [T, E_global]
    at::Tensor bias   = routing_bias.to(at::kFloat).reshape({-1}); // [E_global]

    at::Tensor s            = at::sigmoid(logits);              // [T, E]
    at::Tensor s_with_bias  = s + bias;                         // [T, E]

    at::Tensor s_wb_grouped = s_with_bias.view({T, N_GROUP, E_GLOBAL / N_GROUP});
    at::Tensor top2_vals    = std::get<0>(s_wb_grouped.topk(2, /*dim=*/2));
    at::Tensor group_scores = top2_vals.sum(2);                 // [T, N_GROUP]

    at::Tensor group_idx  = std::get<1>(group_scores.topk(TOPK_GROUP, 1));  // [T, 4]
    at::Tensor group_mask = at::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1.0);
    at::Tensor score_mask = group_mask
        .unsqueeze(2)
        .expand({T, N_GROUP, E_GLOBAL / N_GROUP})
        .reshape({T, E_GLOBAL});

    const float neg_inf = -std::numeric_limits<float>::infinity();
    at::Tensor scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf);
    at::Tensor topk_idx = std::get<1>(scores_pruned.topk(TOP_K, 1));  // [T, TOP_K]

    at::Tensor M       = at::zeros_like(s);
    M.scatter_(1, topk_idx, 1.0f);
    at::Tensor weights = s * M;                                 // [T, E]
    at::Tensor wsum    = weights.sum(1, /*keepdim=*/true) + 1e-20f;
    weights = (weights / wsum) * (float)routed_scaling_factor;  // [T, E]

    // ── 3. Expert loop (C++, no Python overhead) ─────────────────────────────

    at::Tensor acc = at::zeros({T, H}, A.options());  // float32 accumulator

    for (int le = 0; le < E_local; ++le) {
        int ge = local_start + le;
        if (ge < 0 || ge >= E_GLOBAL) continue;

        // tokens that selected this expert
        at::Tensor sel_mask = (topk_idx == ge).any(1);  // [T] bool
        if (!sel_mask.any().item<bool>()) continue;

        at::Tensor tok_idx = sel_mask.nonzero().squeeze(1);  // [Tk]

        at::Tensor A_e   = A.index_select(0, tok_idx);       // [Tk, H]
        at::Tensor W13_e = W13[le];                           // [2I, H]
        at::Tensor W2_e  = W2[le];                            // [H, I]

        at::Tensor G1 = at::mm(A_e, W13_e.t());              // [Tk, 2I]
        at::Tensor X1 = G1.slice(1, 0, I);                   // [Tk, I]
        at::Tensor X2 = G1.slice(1, I, 2 * I);               // [Tk, I]
        at::Tensor C  = at::silu(X2) * X1;                    // [Tk, I]

        at::Tensor O = at::mm(C, W2_e.t());                   // [Tk, H]

        at::Tensor w_tok = weights.index_select(0, tok_idx).select(1, ge)
                               .unsqueeze(1);                 // [Tk, 1]
        acc.index_add_(0, tok_idx, O * w_tok);
    }

    // ── 4. Write output ───────────────────────────────────────────────────────
    output.copy_(acc.to(at::kBFloat16));
}

// ---------------------------------------------------------------------------
// PyBind11 binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel,
          "MoE FP8 block-scale DS routing (CUDA, DPS=true)");
}
