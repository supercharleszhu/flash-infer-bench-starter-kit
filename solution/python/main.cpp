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
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
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

    auto d_offsets = torch::empty({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(dev));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets.data_ptr<int>(), h_offsets.data(),
               sizeof(int)*NUM_LOCAL_EXPERTS, cudaMemcpyHostToDevice, stream));
    auto token_ids = torch::empty({total_assign}, torch::dtype(torch::kInt32).device(dev));
    auto token_wts = torch::empty({total_assign}, torch::dtype(torch::kFloat32).device(dev));
    launch_fill_local_assignments(
        topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(),
        (int)T, (int)local_expert_offset,
        d_offsets.data_ptr<int>(), token_ids.data_ptr<int>(), token_wts.data_ptr<float>(),
        stream);

    // ── 3. Build padded layout and gather in ONE pass (no per-expert loops) ──
    // Pad each expert's count to a multiple of 4 (CUTLASS TMA requirement)
    std::vector<int> h_padded_counts(NUM_LOCAL_EXPERTS);
    std::vector<int> h_padded_offsets(NUM_LOCAL_EXPERTS + 1, 0);
    for (int i = 0; i < NUM_LOCAL_EXPERTS; i++) {
        h_padded_counts[i] = ((h_counts[i] + 3) / 4) * 4;
        h_padded_offsets[i + 1] = h_padded_offsets[i] + h_padded_counts[i];
    }
    int padded_total = h_padded_offsets[NUM_LOCAL_EXPERTS];

    // Build padded_token_ids [padded_total] on CPU: real positions get the
    // original token id, padding positions get T (out-of-range sentinel).
    // gather_fp8_rows writes zero when src_row >= T, so padding rows are zero.
    // Also build padded_safe_ids (0 for padding) for SFA gather via index_select.
    auto padded_ids_cpu = torch::full({padded_total}, (int)T, torch::kInt32);
    auto padded_safe_cpu = torch::zeros({padded_total}, torch::kInt32);
    auto padded_valid_cpu = torch::zeros({padded_total}, torch::kFloat32);
    {
        auto tid_cpu = token_ids.cpu();
        auto tid = tid_cpu.data_ptr<int>();
        auto pid = padded_ids_cpu.data_ptr<int>();
        auto psafe = padded_safe_cpu.data_ptr<int>();
        auto pval = padded_valid_cpu.data_ptr<float>();
        for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
            int Tk = h_counts[e];
            for (int j = 0; j < Tk; j++) {
                int pos = h_padded_offsets[e] + j;
                pid[pos]   = tid[h_offsets[e] + j];
                psafe[pos] = tid[h_offsets[e] + j];
                pval[pos]  = 1.0f;
            }
        }
    }
    auto padded_token_ids = padded_ids_cpu.to(dev);
    auto padded_safe_ids  = padded_safe_cpu.to(dev);
    auto padded_valid     = padded_valid_cpu.to(dev);

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

    // m_indptr: [NUM_LOCAL_EXPERTS+1] int32, padded cumulative offsets
    auto m_indptr_cpu = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::kInt32);
    for (int i = 0; i <= NUM_LOCAL_EXPERTS; i++)
        m_indptr_cpu[i] = h_padded_offsets[i];
    auto m_indptr = m_indptr_cpu.to(dev);

    // ── 5. GEMM1: FP8 blockwise grouped GEMM ──
    auto gemm1_out = torch::empty({padded_total, GEMM1_OUT_SIZE},
                                    torch::dtype(torch::kFloat32).device(dev));

    std::vector<int> active_experts;
    for (int e = 0; e < NUM_LOCAL_EXPERTS; e++)
        if (h_counts[e] > 0) active_experts.push_back(e);

    // ── W2 dequant on separate stream (overlaps with FP8 GEMM1) ──
    cudaStream_t dequant_stream;
    cudaStreamCreate(&dequant_stream);
    cudaEvent_t ready_event;
    cudaEventCreate(&ready_event);
    cudaEventRecord(ready_event, stream);
    cudaStreamWaitEvent(dequant_stream, ready_event, 0);

    auto w2_f32 = torch::zeros({NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE},
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
    cudaEvent_t dequant_done;
    cudaEventCreate(&dequant_done);
    cudaEventRecord(dequant_done, dequant_stream);
    cudaStreamWaitEvent(stream, dequant_done, 0);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(dequant_done);
    cudaStreamDestroy(dequant_stream);

    // ── 8. GEMM2: TF32 grouped GEMM using padded layout ──
    auto expert_offsets_i64 = torch::empty({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt64).device(dev));
    {
        auto eo_cpu = torch::empty({NUM_LOCAL_EXPERTS}, torch::kInt64);
        for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) eo_cpu[e] = h_padded_offsets[e];
        expert_offsets_i64.copy_(eo_cpu);
    }
    auto problem_sizes2 = torch::empty({NUM_LOCAL_EXPERTS, 3}, torch::dtype(torch::kInt32).device(dev));
    {
        auto ps2_cpu = torch::empty({NUM_LOCAL_EXPERTS, 3}, torch::kInt32);
        for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
            ps2_cpu[e][0] = h_padded_counts[e];  // padded M
            ps2_cpu[e][1] = HIDDEN_SIZE;
            ps2_cpu[e][2] = INTERMEDIATE_SIZE;
        }
        problem_sizes2.copy_(ps2_cpu);
    }

    auto gemm2_f32_out = torch::zeros({padded_total, HIDDEN_SIZE},
                                       torch::dtype(torch::kFloat32).device(dev));
    cutlass_moe_gemm_tf32(
        gemm2_f32_out, swiglu_f32, w2_f32,
        expert_offsets_i64, problem_sizes2, NUM_LOCAL_EXPERTS
    );

    // ── 9. Weighted accumulate (single atomic kernel over padded layout) ──
    // padded_token_ids[i] = original token index for position i (T = padding).
    // padded_token_wts[i]: set padded_valid × token_wts below to get 0 for padding.
    auto padded_token_wts_cpu = torch::zeros({padded_total}, torch::kFloat32);
    {
        auto twt_cpu = token_wts.cpu();
        auto twt = twt_cpu.data_ptr<float>();
        auto pwt = padded_token_wts_cpu.data_ptr<float>();
        for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
            int Tk = h_counts[e];
            for (int j = 0; j < Tk; j++) {
                pwt[h_padded_offsets[e] + j] = twt[h_offsets[e] + j];
            }
        }
    }
    auto padded_token_wts = padded_token_wts_cpu.to(dev);

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
