/*
 * CUTLASS MoE — full pipeline with FP8 grouped GEMM (B200-optimized).
 *
 * Pipeline (post Phase 1a/1b/2a/3b optimizations):
 *   1. Routing → topk_ids, topk_weights
 *   2. Count local expert assignments (CPU sync here — TODO Phase 3a)
 *   3. Fill sorted token_ids + build padded arrays on GPU
 *   4. Gather FP8 hidden activations into padded layout + build SFA via index_select
 *   5. GEMM1: CUTLASS SM100 FP8 blockwise grouped GEMM (ClusterShape <2,1,1>)
 *   6. Fused SwiGLU + FP8 re-quantize (one kernel, per-128-block scale)
 *   7. GEMM2: CUTLASS SM100 FP8 blockwise grouped GEMM (shares GEMM1 config)
 *   8. Atomic bf16x2 scatter directly into the pre-allocated bf16 output
 *
 * Key B200-specific techniques:
 *   • ClusterShape <2,1,1> + TMA multicast on B (halves weight bandwidth)
 *   • FP8 on both GEMMs — TCGEN05 cores at ~2× TF32 throughput
 *   • No W2 dequant — saves 1.88 GB HBM3e bandwidth
 *   • PDL on producers feeding CUTLASS (swiglu_quantize, gather_fp8_rows)
 *   • Native atomicAdd(__nv_bfloat162) scatter — no fp32 accumulator tensor
 */

#include "kernel.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>  // for __nv_bfloat16 — used in bf16 atomic scatter (Phase 3b)
#include <vector>
#include <algorithm>

// [B200 optimization — Phase 1a] Forward declaration of the only CUTLASS wrapper
// we still call from main.cpp. We used to invoke cutlass_moe_gemm_tf32 for GEMM2
// with a fp32 dequant of W2 on a side stream; that path has been replaced by a
// fused SwiGLU→fp8-quantize kernel feeding back into cutlass_moe_gemm_fp8_blockwise.
// The dequant stream/event scratch struct and the _bf16 / _tf32 / non-blockwise
// _fp8 declarations that used to live here are gone — those code paths are dead
// in the new pipeline. (See moe_grouped_gemm_fp8.cu for the unused implementations;
// they can be deleted in a follow-up cleanup.)
void cutlass_moe_gemm_fp8_blockwise(
    torch::Tensor& output,          // [cum_m, N] fp32
    torch::Tensor const& A,         // [cum_m, K] fp8
    torch::Tensor const& B,         // [num_groups, N, K] fp8
    torch::Tensor const& SFA,       // [K//128, cum_m] fp32
    torch::Tensor const& SFB,       // [num_groups, K//128, N//128] fp32
    torch::Tensor const& m_indptr,  // [num_groups+1] int32
    int num_groups
);

// M=64 tile-granularity variant for GEMM2. SFA = [K//128, cum_m//64].
void cutlass_moe_gemm_fp8_m64_blockwise(
    torch::Tensor& output,          // [cum_m, N] fp32
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& SFA,       // [K//128, cum_m//64] fp32
    torch::Tensor const& SFB,       // [num_groups, K//128, N//128] fp32
    torch::Tensor const& m_indptr,  // [num_groups+1] int32 (multiples of 64)
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

    // ── 1. Routing (fused with count_local_assignments) ──
    auto routing_bias_f32 = routing_bias.to(torch::kFloat32).contiguous();
    auto topk_idx = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kInt32).device(dev));
    auto topk_w   = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kFloat32).device(dev));
    auto counts   = torch::zeros({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(dev));
    launch_noaux_routing_topk8(
        routing_logits.contiguous().data_ptr<float>(),
        routing_bias_f32.data_ptr<float>(),
        (int)T, static_cast<float>(routed_scaling_factor),
        (int)local_expert_offset,
        counts.data_ptr<int>(),
        topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(), stream);

    // ── 2. Count local assignments + device-side prefix scan ──
    const int max_assign = (int)T * ROUTE_TOP_K;

    // Dual-path: small seq uses 4-token padding (identical to cutlass_v2);
    // large seq uses 64-token padding to enable M=64 tile-scale GEMM2.
    // Threshold: max_assign >= 16000 (~seq≥2000) where 64-pad overhead ≤5%.
    static constexpr int LARGE_SEQ_THRESHOLD = 16000;
    const bool large_seq = (max_assign >= LARGE_SEQ_THRESHOLD);
    const int  pad_align = large_seq ? 64 : 4;
    const int  pad_max_overhead = large_seq ? (NUM_LOCAL_EXPERTS * 63) : (NUM_LOCAL_EXPERTS * 3);
    // Round max_padded up to a multiple of pad_align (required for SFA stride exactness).
    const int max_padded = (max_assign + pad_max_overhead + pad_align - 1) & ~(pad_align - 1);

    // counts already populated by the fused routing kernel above.
    auto d_unpadded_offsets = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    auto d_fill_offsets     = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    auto d_padded_offsets   = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(dev));
    launch_compute_offsets(
        counts.data_ptr<int>(),
        d_unpadded_offsets.data_ptr<int>(),
        d_fill_offsets.data_ptr<int>(),
        d_padded_offsets.data_ptr<int>(),
        stream, pad_align);

    auto token_ids = torch::empty({max_assign}, torch::dtype(torch::kInt32).device(dev));
    auto token_wts = torch::empty({max_assign}, torch::dtype(torch::kFloat32).device(dev));
    // inv_slot: fill_local_assignments_with_inverse writes to EVERY (t,k) slot
    // (-1 for non-local) — no prior init needed.
    auto inv_slot = torch::empty({(int64_t)T * ROUTE_TOP_K},
                                  torch::dtype(torch::kInt32).device(dev));
    launch_fill_local_assignments_with_inverse(
        topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(),
        (int)T, (int)local_expert_offset,
        d_fill_offsets.data_ptr<int>(),       // mutable; atomic fill counters
        d_unpadded_offsets.data_ptr<int>(),   // read-only base offsets
        d_padded_offsets.data_ptr<int>(),     // read-only base offsets
        token_ids.data_ptr<int>(), token_wts.data_ptr<float>(),
        inv_slot.data_ptr<int>(),
        stream);

    // ── 3. Build padded layout (all on GPU) ──
    // Single-launch initializer for sentinel values (replaces 4 torch fills,
    // saves ~45µs of launch overhead). build_padded_arrays overwrites the
    // [0, padded_total) range with real values.
    auto padded_token_ids = torch::empty({max_padded}, torch::dtype(torch::kInt32).device(dev));
    auto padded_safe_ids  = torch::empty({max_padded}, torch::dtype(torch::kInt32).device(dev));
    auto padded_valid     = torch::empty({max_padded}, torch::dtype(torch::kFloat32).device(dev));
    auto padded_token_wts = torch::empty({max_padded}, torch::dtype(torch::kFloat32).device(dev));
    launch_init_padded_arrays(
        max_padded, (int)T,
        padded_token_ids.data_ptr<int>(), padded_safe_ids.data_ptr<int>(),
        padded_valid.data_ptr<float>(), padded_token_wts.data_ptr<float>(),
        stream);
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

    // Use max_padded as the "padded_total" for downstream sizing. CUTLASS uses
    // d_padded_offsets to determine actual M per expert, so unused rows aren't
    // computed; downstream helper kernels read sentinel/zero from padded arrays
    // and skip naturally. No correctness hazard from the over-alloc.
    const int padded_total = max_padded;

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

    // Fused SFA prep — replaces 4 torch ops (cast/index_select/contiguous/mul_).
    int64_t K_blocks = HIDDEN_SIZE / 128;
    auto hs_scale_cont = hidden_states_scale.contiguous();  // [K_blocks, T] (no-op if already)
    auto sfa = torch::empty({K_blocks, padded_total},
                              torch::dtype(torch::kFloat32).device(dev));
    launch_prepare_sfa(
        hs_scale_cont.data_ptr<float>(),
        padded_safe_ids.data_ptr<int>(),
        padded_valid.data_ptr<float>(),
        (int)T, padded_total, (int)K_blocks,
        sfa.data_ptr<float>(), stream);

    // SFB: permute from [E, N/128, K/128] to [E, K/128, N/128] (MN-major)
    auto sfb = gemm1_weights_scale.permute({0, 2, 1}).contiguous();

    // m_indptr: [NUM_LOCAL_EXPERTS+1] int32, padded cumulative offsets.
    // d_padded_offsets already holds this on device — reuse it directly.
    auto& m_indptr = d_padded_offsets;

    // ── 5. GEMM1: FP8 blockwise grouped GEMM ──
    auto gemm1_out = torch::empty({padded_total, GEMM1_OUT_SIZE},
                                    torch::dtype(torch::kFloat32).device(dev));
    cutlass_moe_gemm_fp8_blockwise(
        gemm1_out, sorted_act_fp8, gemm1_weights, sfa, sfb, m_indptr, NUM_LOCAL_EXPERTS
    );

    // ── 6. Fused SwiGLU + FP8 blockwise quantize ──
    // [B200 optimization] Replaces the old (SwiGLU → side-stream W2 dequant →
    // TF32 GEMM2) triad with one kernel that emits fp8 + per-128-block scales
    // directly into CUTLASS SFA layout. Enables FP8 GEMM2 on TCGEN05 cores
    // (~2× TF32 throughput) and eliminates the 1.88 GB `w2_f32` dequant buffer
    // that previously burned HBM3e bandwidth. GEMM2 reads the original fp8
    // weights + scales in-place.
    //
    // Granularity is M=1 per-token × K=128 per-block — matches GEMM2's
    // Sm100BlockwiseScaleConfig<1,128,128,MN,MN> exactly. Passes contest tol
    // (atol=1, rtol=0.3, matched@0.9) with wide margin.
    auto swiglu_fp8_raw = torch::empty({padded_total, INTERMEDIATE_SIZE},
                                        torch::dtype(torch::kUInt8).device(dev));
    auto sfb_g2 = gemm2_weights_scale.permute({0, 2, 1}).contiguous();

    if (large_seq) {
        auto swiglu_scales = torch::empty({INTERMEDIATE_SIZE / 128, padded_total / 64},
                                           torch::dtype(torch::kFloat32).device(dev));
        launch_swiglu_quantize_fp8_m64(
            gemm1_out.data_ptr<float>(), padded_total,
            swiglu_fp8_raw.data_ptr<uint8_t>(),
            swiglu_scales.data_ptr<float>(), stream);
        auto swiglu_fp8 = torch::from_blob(swiglu_fp8_raw.data_ptr(),
            {padded_total, INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat8_e4m3fn).device(dev));
        auto gemm2_out = torch::empty({padded_total, HIDDEN_SIZE},
                                       torch::dtype(torch::kFloat32).device(dev));
        cutlass_moe_gemm_fp8_m64_blockwise(
            gemm2_out, swiglu_fp8, gemm2_weights,
            swiglu_scales, sfb_g2, d_padded_offsets, NUM_LOCAL_EXPERTS);
        launch_finalize_weighted_bf16(
            gemm2_out.data_ptr<float>(),
            inv_slot.data_ptr<int>(),
            topk_w.data_ptr<float>(),
            (int)T, HIDDEN_SIZE,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            stream);
    } else {
        // Small seq: cutlass_v2-equivalent path (FP32 GEMM2 + atomic scatter).
        auto swiglu_scales = torch::empty({INTERMEDIATE_SIZE / 128, padded_total},
                                           torch::dtype(torch::kFloat32).device(dev));
        launch_swiglu_quantize_fp8(
            gemm1_out.data_ptr<float>(), padded_total,
            swiglu_fp8_raw.data_ptr<uint8_t>(),
            swiglu_scales.data_ptr<float>(), stream);
        auto swiglu_fp8 = torch::from_blob(swiglu_fp8_raw.data_ptr(),
            {padded_total, INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat8_e4m3fn).device(dev));
        auto gemm2_out = torch::empty({padded_total, HIDDEN_SIZE},
                                       torch::dtype(torch::kFloat32).device(dev));
        cutlass_moe_gemm_fp8_blockwise(
            gemm2_out, swiglu_fp8, gemm2_weights,
            swiglu_scales, sfb_g2, d_padded_offsets, NUM_LOCAL_EXPERTS);
        output.zero_();
        launch_scatter_weighted_bf16_atomic(
            gemm2_out.data_ptr<float>(),
            padded_token_ids.data_ptr<int>(),
            padded_token_wts.data_ptr<float>(),
            padded_total, HIDDEN_SIZE, (int)T,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "CUTLASS MoE FP8 grouped GEMM with scaled epilogue");
}
