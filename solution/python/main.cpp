/*
 * CUTLASS MoE — full pipeline with FP8 grouped GEMM + scaled epilogue.
 *
 * Pipeline:
 *   1. Routing → topk_ids, topk_weights
 *   2. Count/fill local expert assignments → sorted token_ids, offsets
 *   3. Gather FP8 activations into flat sorted layout [total_tokens, K]
 *   4. Convert block scales → per-token (act) and per-channel (weight)
 *   5. GEMM1: CUTLASS FP8 grouped GEMM with scale epilogue
 *   6. SwiGLU (fused CUDA kernel on sorted output)
 *   7. Quantize SwiGLU output to FP8 + compute per-token scale
 *   8. GEMM2: CUTLASS FP8 grouped GEMM with scale epilogue
 *   9. Weighted accumulate back to [T, H]
 */

#include "kernel.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <algorithm>
#include <unordered_map>

// Cached per-device dequant stream and sync events. Lazily created on first
// use, never destroyed (process lifetime). Replaces the per-call
// cudaStreamCreate / cudaEventCreate / destroy pairs (~100μs aggregate).
// Events use cudaEventDisableTiming since they are sync-only.
struct DequantScratch {
    cudaStream_t stream;
    cudaEvent_t  ready;       // main-stream → dequant-stream handshake
    cudaEvent_t  done;        // dequant-stream → main-stream handshake
};

static DequantScratch& get_dequant_scratch(int device_id) {
    static std::unordered_map<int, DequantScratch> cache;
    auto it = cache.find(device_id);
    if (it != cache.end()) return it->second;
    DequantScratch s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.done,  cudaEventDisableTiming));
    cache[device_id] = s;
    return cache[device_id];
}

// Forward declarations (defined in moe_grouped_gemm_fp8.cu)
void cutlass_moe_gemm_bf16(
    torch::Tensor& output,
    torch::Tensor const& activations,
    torch::Tensor const& weights,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    int num_experts
);

void cutlass_moe_gemm_tf32(
    torch::Tensor& output,
    torch::Tensor const& activations,
    torch::Tensor const& weights,
    torch::Tensor const& padded_offsets,  // [E+1] int32 cumulative
    int num_experts
);

void cutlass_moe_gemm_fp8(
    torch::Tensor& output,
    torch::Tensor const& activations,
    torch::Tensor const& weights,
    torch::Tensor const& a_scales,
    torch::Tensor const& a_scale_offsets,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    int num_experts
);

void cutlass_moe_gemm_fp8_blockwise(
    torch::Tensor& output,          // [cum_m, N] fp32
    torch::Tensor const& A,         // [cum_m, K] fp8
    torch::Tensor const& B,         // [num_groups, N, K] fp8
    torch::Tensor const& SFA,       // [K//128, cum_m] fp32
    torch::Tensor const& SFB,       // [num_groups, K//128, N//128] fp32
    torch::Tensor const& m_indptr,  // [num_groups+1] int32
    int num_groups
);

// (No per-row scale conversion needed — blockwise CUTLASS handles per-K-block scales directly)

// ═══════════════════════════════════════════════════════════════════════════════
// Main entry
// ═══════════════════════════════════════════════════════════════════════════════

void run(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,         // [T, H] fp8
    torch::Tensor hidden_states_scale,   // [H/128, T] float32
    torch::Tensor gemm1_weights,         // [E, 2I, H] fp8
    torch::Tensor gemm1_weights_scale,   // [E, 2I/128, H/128] float32
    torch::Tensor gemm2_weights,         // [E, H, I] fp8
    torch::Tensor gemm2_weights_scale,   // [E, H/128, I/128] float32
    int64_t local_expert_offset,
    double routed_scaling_factor,
    torch::Tensor output                 // [T, H] bfloat16 (DPS, pre-allocated)
) {
    TORCH_CHECK(routing_logits.is_cuda() && hidden_states.is_cuda());
    const int64_t T = routing_logits.size(0);
    c10::cuda::CUDAGuard device_guard(routing_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto dev = routing_logits.device();

    // ── 1. Routing ──
    auto routing_bias_f32 = routing_bias.to(torch::kFloat32).contiguous();
    auto topk_idx = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kInt32).device(dev));
    auto topk_w = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kFloat32).device(dev));
    launch_noaux_routing_topk8(
        routing_logits.contiguous().data_ptr<float>(),
        routing_bias_f32.data_ptr<float>(),
        (int)T, static_cast<float>(routed_scaling_factor),
        topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(), stream);

    // ── 2. Count + fill local assignments ──
    auto counts = torch::zeros({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(dev));
    launch_count_local_assignments(
        topk_idx.data_ptr<int>(), (int)T, (int)local_expert_offset,
        counts.data_ptr<int>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto counts_cpu = counts.cpu();
    auto cp = counts_cpu.data_ptr<int>();
    std::vector<int> h_counts(NUM_LOCAL_EXPERTS);
    int total_assign = 0;
    for (int i = 0; i < NUM_LOCAL_EXPERTS; i++) {
        h_counts[i] = cp[i];
        total_assign += h_counts[i];
    }
    std::vector<int> h_offsets(NUM_LOCAL_EXPERTS + 1, 0);
    for (int i = 0; i < NUM_LOCAL_EXPERTS; i++) h_offsets[i+1] = h_offsets[i] + h_counts[i];

    if (total_assign == 0) {
        output.zero_();
        return;
    }

    // Upload unpadded cumulative offsets [E+1] to device. We need TWO copies:
    //   d_fill_offsets:      mutated in-place by fill_local_assignments (atomics)
    //   d_unpadded_offsets:  read-only copy for build_padded_arrays below
    std::vector<int> h_offsets_int(NUM_LOCAL_EXPERTS + 1, 0);
    for (int i = 0; i <= NUM_LOCAL_EXPERTS; i++) h_offsets_int[i] = h_offsets[i];
    auto d_fill_offsets = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    auto d_unpadded_offsets = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    CUDA_CHECK(cudaMemcpyAsync(d_fill_offsets.data_ptr<int>(), h_offsets_int.data(),
               sizeof(int)*(NUM_LOCAL_EXPERTS+1), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_unpadded_offsets.data_ptr<int>(), h_offsets_int.data(),
               sizeof(int)*(NUM_LOCAL_EXPERTS+1), cudaMemcpyHostToDevice, stream));

    auto token_ids = torch::empty({total_assign}, torch::dtype(torch::kInt32).device(dev));
    auto token_wts = torch::empty({total_assign}, torch::dtype(torch::kFloat32).device(dev));
    launch_fill_local_assignments(
        topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(),
        (int)T, (int)local_expert_offset,
        d_fill_offsets.data_ptr<int>(),  // mutable; gets incremented atomically
        token_ids.data_ptr<int>(), token_wts.data_ptr<float>(),
        stream);

    // ── 3. Build padded layout (offsets on CPU cheap; padded arrays on GPU) ──
    std::vector<int> h_padded_counts(NUM_LOCAL_EXPERTS);
    std::vector<int> h_padded_offsets(NUM_LOCAL_EXPERTS + 1, 0);
    for (int i = 0; i < NUM_LOCAL_EXPERTS; i++) {
        h_padded_counts[i] = ((h_counts[i] + 3) / 4) * 4;
        h_padded_offsets[i + 1] = h_padded_offsets[i] + h_padded_counts[i];
    }
    int padded_total = h_padded_offsets[NUM_LOCAL_EXPERTS];

    auto d_padded_offsets = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    CUDA_CHECK(cudaMemcpyAsync(d_padded_offsets.data_ptr<int>(), h_padded_offsets.data(),
               sizeof(int)*(NUM_LOCAL_EXPERTS+1), cudaMemcpyHostToDevice, stream));

    // Build padded arrays on GPU (no token_ids.cpu() sync needed).
    auto padded_token_ids = torch::empty({padded_total}, torch::dtype(torch::kInt32).device(dev));
    auto padded_safe_ids  = torch::empty({padded_total}, torch::dtype(torch::kInt32).device(dev));
    auto padded_valid     = torch::empty({padded_total}, torch::dtype(torch::kFloat32).device(dev));
    auto padded_token_wts = torch::empty({padded_total}, torch::dtype(torch::kFloat32).device(dev));
    launch_build_padded_arrays(
        d_unpadded_offsets.data_ptr<int>(),
        d_padded_offsets.data_ptr<int>(),
        token_ids.data_ptr<int>(),
        token_wts.data_ptr<float>(),
        (int)T, NUM_LOCAL_EXPERTS,
        padded_token_ids.data_ptr<int>(),
        padded_safe_ids.data_ptr<int>(),
        padded_valid.data_ptr<float>(),
        padded_token_wts.data_ptr<float>(),
        stream);

    // Single gather: sentinel=T makes padding rows zero in the kernel.
    auto sorted_act_fp8_raw = torch::empty({padded_total, HIDDEN_SIZE},
                                            torch::dtype(torch::kUInt8).device(dev));
    launch_gather_fp8_rows(
        reinterpret_cast<const uint8_t*>(hidden_states.data_ptr()),
        padded_token_ids.data_ptr<int>(),
        (int)T, padded_total, HIDDEN_SIZE,
        sorted_act_fp8_raw.data_ptr<uint8_t>(), stream);
    auto sorted_act_fp8 = torch::from_blob(sorted_act_fp8_raw.data_ptr(),
        {padded_total, HIDDEN_SIZE}, torch::dtype(torch::kFloat8_e4m3fn).device(dev));

    // Single SFA gather via index_select using safe ids, mask zero padding cols.
    int64_t K_blocks = HIDDEN_SIZE / 128;
    auto hs_scale_cont = hidden_states_scale.contiguous();  // [K_blocks, T]
    auto sfa = hs_scale_cont.index_select(1, padded_safe_ids.to(torch::kLong)).contiguous();
    sfa.mul_(padded_valid.unsqueeze(0));  // zero padding columns

    // SFB: permute from [E, N/128, K/128] to [E, K/128, N/128] (MN-major)
    auto sfb = gemm1_weights_scale.permute({0, 2, 1}).contiguous();

    // m_indptr: [NUM_LOCAL_EXPERTS+1] int32, padded cumulative offsets.
    // d_padded_offsets already holds this on device — reuse it directly.
    auto& m_indptr = d_padded_offsets;

    // ── 5. GEMM1: FP8 blockwise grouped GEMM ──
    auto gemm1_out = torch::empty({padded_total, GEMM1_OUT_SIZE},
                                    torch::dtype(torch::kFloat32).device(dev));

    std::vector<int> active_experts;
    for (int e = 0; e < NUM_LOCAL_EXPERTS; e++)
        if (h_counts[e] > 0) active_experts.push_back(e);

    // ── W2 dequant on separate stream (overlaps with FP8 GEMM1) ──
    // Reuse cached stream + events; per-call create/destroy costs ~100μs total.
    auto& scratch = get_dequant_scratch(dev.index());
    cudaStream_t dequant_stream = scratch.stream;
    cudaEventRecord(scratch.ready, stream);
    cudaStreamWaitEvent(dequant_stream, scratch.ready, 0);

    // empty (not zeros): launch_fused_dequant_experts writes all active-expert
    // slices; inactive experts have M_e=0 so CUTLASS never reads their weights.
    auto w2_f32 = torch::empty({NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE},
                                torch::dtype(torch::kFloat32).device(dev));
    if (!active_experts.empty()) {
        auto d_active = torch::tensor(active_experts, torch::dtype(torch::kInt32).device(dev));
        launch_fused_dequant_experts(
            reinterpret_cast<const uint8_t*>(gemm2_weights.data_ptr()),
            gemm2_weights_scale.contiguous().data_ptr<float>(),
            w2_f32.data_ptr<float>(),
            d_active.data_ptr<int>(),
            (int)active_experts.size(), HIDDEN_SIZE, INTERMEDIATE_SIZE, dequant_stream);
    }

    cutlass_moe_gemm_fp8_blockwise(
        gemm1_out, sorted_act_fp8, gemm1_weights, sfa, sfb, m_indptr, NUM_LOCAL_EXPERTS
    );

    // ── 6. SwiGLU on padded output (fused CUDA kernel) ──
    auto& g1_f32 = gemm1_out;
    auto swiglu_f32 = torch::empty({padded_total, INTERMEDIATE_SIZE},
                                     torch::dtype(torch::kFloat32).device(dev));
    launch_swiglu(g1_f32.data_ptr<float>(), padded_total,
                  swiglu_f32.data_ptr<float>(), stream);

    // ── Wait for W2 dequant to finish before GEMM2 ──
    cudaEventRecord(scratch.done, dequant_stream);
    cudaStreamWaitEvent(stream, scratch.done, 0);
    // stream + events are cached; do not destroy

    // ── 8. GEMM2: TF32 grouped GEMM using padded layout ──
    // Pass d_padded_offsets directly — cutlass_moe_gemm_tf32 now builds pointer
    // and problem-shape arrays on device from the int32 cumulative offsets.
    // empty (not zeros): GEMM2 beta=0 overwrites every active row; padding rows
    // have token_wts=0 in the downstream accumulate so their value is unused.
    auto gemm2_f32_out = torch::empty({padded_total, HIDDEN_SIZE},
                                       torch::dtype(torch::kFloat32).device(dev));
    cutlass_moe_gemm_tf32(
        gemm2_f32_out, swiglu_f32, w2_f32,
        d_padded_offsets, NUM_LOCAL_EXPERTS
    );

    // ── 9. Weighted accumulate (single atomic kernel over padded layout) ──
    // padded_token_wts is already built on GPU by launch_build_padded_arrays.
    auto output_f32 = torch::zeros({T, HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(dev));
    launch_accumulate_weighted_add_atomic(
        gemm2_f32_out.data_ptr<float>(),
        padded_token_ids.data_ptr<int>(),
        padded_token_wts.data_ptr<float>(),
        padded_total, HIDDEN_SIZE, (int)T,
        output_f32.data_ptr<float>(), stream);

    // Write directly into the pre-allocated DPS output tensor (bf16).
    output.copy_(output_f32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "CUTLASS MoE FP8 grouped GEMM with scaled epilogue");
}
