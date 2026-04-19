/*
 * CUTLASS FP8 Grouped GEMM with per-K-block scaling (Blackwell SM100).
 *
 * Uses Sm100BlockwiseScaleConfig to integrate per-block scale factors
 * directly into the GEMM mainloop — no epilogue scaling needed.
 *
 * Scale granularity: M=1 (per-token for A), N=128, K=128 (per-block for both).
 * This exactly matches DeepSeek-V3's FP8 block-scale quantization format.
 *
 * Ported from SM90 (Hopper) to SM100 (Blackwell) using CUTLASS 4.4 SM100 schedules.
 */

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "kernel.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using namespace cute;

// ═══════════════════════════════════════════════════════════════════════════════
// Types and config
// ═══════════════════════════════════════════════════════════════════════════════

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using ElementAB = cutlass::float_e4m3_t;
using ElementD  = float;
using ElementC  = float;
using ElementAccumulator = float;
using ElementBlockScale  = float;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag       = cutlass::arch::Sm100;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value; // 4 for float

// Blockwise scale granularity — matches DeepSeek-V3:
//   A scales: per-token (M=1) × per-128-element K-block
//   B scales: per-128-element N-block × per-128-element K-block
// SM100 blockwise scale granularity — M=128 (per-tile), N=128, K=128
// SM100 TCGEN05 doesn't support per-token (M=1) scale granularity.
// Per-token A scales are pre-applied to activations before the GEMM.
using MmaTileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ScaleConfig = decltype(cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// ═══════════════════════════════════════════════════════════════════════════════
// SM100 FP8 Blockwise Grouped GEMM (GEMM1)
// Uses 2SM cooperative MMA with TCGEN05 tensor cores
// ═══════════════════════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;  // 1SM (matches CUTLASS example 81)

using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;  // TMA epilogue (matches example 81)
using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentD,
    ElementD, LayoutD*, AlignmentD,
    EpilogueSchedule, FusionOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementAB, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentAB,
    ElementAB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentAB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// ═══════════════════════════════════════════════════════════════════════════════
// SM100 TF32 Grouped GEMM for GEMM2 (FP32 A × FP32-dequanted B → FP32)
// ═══════════════════════════════════════════════════════════════════════════════

using TF32_TileShape    = Shape<_128, _128, _32>;  // K=32 for TF32
using TF32_ClusterShape = Shape<_1, _1, _1>;

using TF32_EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm;
using TF32_FusionOp = cutlass::epilogue::fusion::LinearCombination<float, float>;

using TF32_CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TF32_TileShape, TF32_ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,          // accumulator, compute
    float, cutlass::layout::RowMajor*, 4,  // C
    float, cutlass::layout::RowMajor*, 4,  // D
    TF32_EpilogueSchedule, TF32_FusionOp
>::CollectiveOp;

using TF32_KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;

using TF32_CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    float, cutlass::layout::RowMajor*, 4,   // A: FP32 row-major (auto TF32)
    float, cutlass::layout::ColumnMajor*, 4, // B: FP32 col-major (auto TF32)
    float,                                    // accumulator
    TF32_TileShape, TF32_ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename TF32_CollectiveEpilogue::SharedStorage))>,
    TF32_KernelSchedule
>::CollectiveOp;

using TF32_GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, TF32_CollectiveMainloop, TF32_CollectiveEpilogue>;
using TF32_Gemm = cutlass::gemm::device::GemmUniversalAdapter<TF32_GemmKernel>;

using TF32_StrideA = typename TF32_Gemm::GemmKernel::InternalStrideA;
using TF32_StrideB = typename TF32_Gemm::GemmKernel::InternalStrideB;
using TF32_StrideC = typename TF32_Gemm::GemmKernel::InternalStrideC;
using TF32_StrideD = typename TF32_Gemm::GemmKernel::InternalStrideD;

// ═══════════════════════════════════════════════════════════════════════════════
// SM100 BF16 Grouped GEMM for GEMM2 (BF16 A × BF16 B → FP32)
// ═══════════════════════════════════════════════════════════════════════════════

using BF16_TileShape    = Shape<_128, _128, _64>;  // K=64 for BF16
using BF16_ClusterShape = Shape<_1, _1, _1>;

using BF16_EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm;
using BF16_FusionOp = cutlass::epilogue::fusion::LinearCombination<float, float>;

using BF16_CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    BF16_TileShape, BF16_ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::RowMajor*, 4,
    float, cutlass::layout::RowMajor*, 4,
    BF16_EpilogueSchedule, BF16_FusionOp
>::CollectiveOp;

using BF16_KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;

using BF16_CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cutlass::bfloat16_t, cutlass::layout::RowMajor*, 8,   // A: BF16
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor*, 8, // B: BF16
    float,
    BF16_TileShape, BF16_ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename BF16_CollectiveEpilogue::SharedStorage))>,
    BF16_KernelSchedule
>::CollectiveOp;

using BF16_GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, BF16_CollectiveMainloop, BF16_CollectiveEpilogue>;
using BF16_Gemm = cutlass::gemm::device::GemmUniversalAdapter<BF16_GemmKernel>;

using BF16_StrideA = typename BF16_Gemm::GemmKernel::InternalStrideA;
using BF16_StrideB = typename BF16_Gemm::GemmKernel::InternalStrideB;
using BF16_StrideC = typename BF16_Gemm::GemmKernel::InternalStrideC;
using BF16_StrideD = typename BF16_Gemm::GemmKernel::InternalStrideD;

// ═══════════════════════════════════════════════════════════════════════════════
// Flashinfer-style FP8 Blockwise Grouped GEMM
// Per-token A scales (M=1), per-128-block B scales, MN-major layout.
// Ported from flashinfer group_gemm_fp8_groupwise_sm100.cuh (CUTLASS v4.4).
// ═══════════════════════════════════════════════════════════════════════════════

namespace fi_gemm {

// Per-token A scale (granularity M=1), 128×128 block B scale, MN-major
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
    1, 128, 128, UMMA::Major::MN, UMMA::Major::MN>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// 1SM tile — safe for all batch sizes (even 1 token per expert)
using FI_TileShape    = Shape<_128, _128, _128>;
using FI_ClusterShape = Shape<_1, _1, _1>;

using FI_KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
using FI_EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// Output fp32 (needed for SwiGLU)
using FI_ElementD = float;
using FI_LayoutD  = cutlass::layout::RowMajor;
static constexpr int FI_AlignmentD = 128 / cutlass::sizeof_bits<FI_ElementD>::value;

using FI_FusionOp = cutlass::epilogue::fusion::LinearCombination<FI_ElementD, ElementAccumulator>;

using FI_CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    FI_TileShape, FI_ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    void, void*, 0,          // no C input
    FI_ElementD, FI_LayoutD*, FI_AlignmentD,
    FI_EpilogueSchedule, FI_FusionOp
>::CollectiveOp;

using FI_CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementAB, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentAB,
    ElementAB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentAB,
    ElementAccumulator,
    FI_TileShape, FI_ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename FI_CollectiveEpilogue::SharedStorage))>,
    FI_KernelSchedule
>::CollectiveOp;

using FI_GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, FI_CollectiveMainloop, FI_CollectiveEpilogue>;
using FI_Gemm = cutlass::gemm::device::GemmUniversalAdapter<FI_GemmKernel>;

using FI_StrideA = typename FI_Gemm::GemmKernel::InternalStrideA;
using FI_StrideB = typename FI_Gemm::GemmKernel::InternalStrideB;
using FI_StrideD = typename FI_Gemm::GemmKernel::InternalStrideD;

}  // namespace fi_gemm

// Pointer setup kernel with Programmatic Dependent Launch (PDL)
__global__ void setup_fi_gemm_args(
    ElementAB* A, ElementAB* B, float* SFA, float* SFB, float* D,
    int* m_indptr, int max_m, int n, int k, int num_groups,
    ProblemShape::UnderlyingProblemShape* problem_sizes,
    const ElementAB** A_ptr, const ElementAB** B_ptr,
    const float** SFA_ptr, const float** SFB_ptr,
    float** D_ptr,
    fi_gemm::FI_StrideA* stride_A, fi_gemm::FI_StrideB* stride_B,
    fi_gemm::FI_StrideD* stride_D,
    fi_gemm::LayoutSFA* layout_SFA, fi_gemm::LayoutSFB* layout_SFB
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_groups) return;

    int sf_n = n / 128;
    int sf_k = k / 128;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
#endif

    int m_offset = m_indptr[i];
    int m_next   = m_indptr[i + 1];
    int m = m_next - m_offset;

    problem_sizes[i] = ProblemShape::UnderlyingProblemShape(m, n, k);
    // RowMajor A [M,K]: leading dim = K
    stride_A[i] = fi_gemm::FI_StrideA{k, cute::Int<1>{}, cute::Int<0>{}};
    // ColumnMajor B [N,K]: leading dim = K
    stride_B[i] = fi_gemm::FI_StrideB{k, cute::Int<1>{}, cute::Int<0>{}};
    // RowMajor D [M,N]: leading dim = N
    stride_D[i] = fi_gemm::FI_StrideD{n, cute::Int<1>{}, cute::Int<0>{}};

    A_ptr[i]   = A   + (int64_t)m_offset * k;
    B_ptr[i]   = B   + (int64_t)i * n * k;
    D_ptr[i]   = D   + (int64_t)m_offset * n;

    // MN-major SFA: [K//128, cum_m] — column offset by m_offset
    layout_SFA[i] = fi_gemm::ScaleConfig::tile_atom_to_shape_SFA(make_shape(max_m, n, k, 1));
    SFA_ptr[i]    = SFA + m_offset;

    // MN-major SFB: [num_groups, K//128, N//128] — expert i block
    layout_SFB[i] = fi_gemm::ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    SFB_ptr[i]    = SFB + (int64_t)i * sf_n * sf_k;
}

#endif // SM100

// ═══════════════════════════════════════════════════════════════════════════════
// Pointer setup kernel (legacy — used by old cutlass_moe_gemm_fp8)
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void setup_expert_pointers_blockwise(
    int64_t* expert_offsets,       // [E] cumulative token offsets
    int64_t* sfa_offsets,          // [E] cumulative SFA scale offsets (pre-computed)
    ElementAB** a_ptrs,            // [E] output
    ElementAB** b_ptrs,            // [E] output
    ElementD** out_ptrs,           // [E] output
    ElementBlockScale** sfa_ptrs,  // [E] output — scale for A
    ElementBlockScale** sfb_ptrs,  // [E] output — scale for B
    ElementAB* a_base,             // flat activations [total_tokens, K]
    ElementAB* b_base,             // stacked weights [E, N, K]
    ElementD* out_base,            // flat output [total_tokens, N]
    ElementBlockScale* sfa_base,   // repacked [K_blocks*M_0 | K_blocks*M_1 | ...] per-expert
    ElementBlockScale* sfb_base,   // [E, N/128, K/128] — B scales
    int64_t N, int64_t K,
    int64_t K_blocks               // K / 128
) {
    int e = threadIdx.x;
    int64_t offset = expert_offsets[e];

    a_ptrs[e] = a_base + offset * K;
    b_ptrs[e] = b_base + (int64_t)e * N * K;
    out_ptrs[e] = out_base + offset * N;

    // A scale: per-expert repacked as [K_blocks, M_e] contiguous blocks
    sfa_ptrs[e] = sfa_base + sfa_offsets[e];

    // B scale: [E, N/128, K/128] contiguous — expert e block
    int64_t N_blocks = N / 128;
    sfb_ptrs[e] = sfb_base + (int64_t)e * N_blocks * K_blocks;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Entry point — FP8 blockwise grouped GEMM (GEMM1)
// ═══════════════════════════════════════════════════════════════════════════════

void cutlass_moe_gemm_fp8(
    torch::Tensor& output,              // [total_tokens, N] fp32
    torch::Tensor const& activations,   // [total_tokens, K] fp8
    torch::Tensor const& weights,       // [E, N, K] fp8
    torch::Tensor const& a_scales,      // repacked per-expert A scales (flat)
    torch::Tensor const& a_scale_offsets, // [E] int64 — per-expert offsets into a_scales
    torch::Tensor const& b_scales,      // [E, N/128, K/128] float32 (blockwise)
    torch::Tensor const& expert_offsets, // [E] int64
    torch::Tensor const& problem_sizes, // [E, 3] int32 (M_i, N, K)
    int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto stream = at::cuda::getCurrentCUDAStream(activations.device().index());
    auto dev = activations.device();
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    int64_t N = output.size(1);
    int64_t K = activations.size(1);
    int64_t K_blocks = K / 128;

    // Per-expert pointer arrays
    torch::Tensor a_ptrs = torch::empty(num_experts, opts_i64);
    torch::Tensor b_ptrs = torch::empty(num_experts, opts_i64);
    torch::Tensor out_ptrs = torch::empty(num_experts, opts_i64);
    torch::Tensor sfa_ptrs = torch::empty(num_experts, opts_i64);
    torch::Tensor sfb_ptrs = torch::empty(num_experts, opts_i64);

    setup_expert_pointers_blockwise<<<1, num_experts, 0, stream>>>(
        static_cast<int64_t*>(expert_offsets.data_ptr()),
        static_cast<int64_t*>(a_scale_offsets.data_ptr()),
        reinterpret_cast<ElementAB**>(a_ptrs.data_ptr()),
        reinterpret_cast<ElementAB**>(b_ptrs.data_ptr()),
        reinterpret_cast<ElementD**>(out_ptrs.data_ptr()),
        reinterpret_cast<ElementBlockScale**>(sfa_ptrs.data_ptr()),
        reinterpret_cast<ElementBlockScale**>(sfb_ptrs.data_ptr()),
        reinterpret_cast<ElementAB*>(activations.data_ptr()),
        reinterpret_cast<ElementAB*>(weights.data_ptr()),
        reinterpret_cast<ElementD*>(output.data_ptr()),
        static_cast<ElementBlockScale*>(a_scales.data_ptr()),
        static_cast<ElementBlockScale*>(b_scales.data_ptr()),
        N, K, K_blocks
    );

    // Strides — 1D arrays, one int64 (leading dim) per expert
    auto strides_a = torch::full({num_experts}, K, opts_i64);
    auto strides_b = torch::full({num_experts}, K, opts_i64);
    auto strides_c = torch::full({num_experts}, N, opts_i64);

    // Scale factor layouts — one LayoutSFA/LayoutSFB per expert
    // Compute on host then copy to device
    std::vector<LayoutSFA> layouts_sfa_host(num_experts);
    std::vector<LayoutSFB> layouts_sfb_host(num_experts);

    auto ps_cpu = problem_sizes.cpu().contiguous();
    auto ps_ptr = ps_cpu.data_ptr<int>();
    for (int e = 0; e < num_experts; e++) {
        int M_e = ps_ptr[e * 3 + 0];
        int N_e = ps_ptr[e * 3 + 1];
        int K_e = ps_ptr[e * 3 + 2];
        layouts_sfa_host[e] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M_e, N_e, K_e, 1));
        layouts_sfb_host[e] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M_e, N_e, K_e, 1));
    }

    // Allocate device arrays for layouts
    auto layouts_sfa_dev = torch::empty({(int64_t)(num_experts * sizeof(LayoutSFA))},
                                         torch::dtype(torch::kByte).device(dev));
    auto layouts_sfb_dev = torch::empty({(int64_t)(num_experts * sizeof(LayoutSFB))},
                                         torch::dtype(torch::kByte).device(dev));
    cudaMemcpyAsync(layouts_sfa_dev.data_ptr(), layouts_sfa_host.data(),
                    num_experts * sizeof(LayoutSFA), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(layouts_sfb_dev.data_ptr(), layouts_sfb_host.data(),
                    num_experts * sizeof(LayoutSFB), cudaMemcpyHostToDevice, stream);

    // Problem shape — pass both device AND host pointers (SM100 CLC scheduler needs host)
    ProblemShape::UnderlyingProblemShape* prob_shapes_dev =
        static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());
    ProblemShape::UnderlyingProblemShape* prob_shapes_host =
        static_cast<ProblemShape::UnderlyingProblemShape*>(ps_cpu.data_ptr());
    ProblemShape prob_shape{num_experts, prob_shapes_dev, prob_shapes_host};

    // Mainloop args — includes blockscale pointers and layouts
    typename Gemm::GemmKernel::MainloopArguments mainloop_args{
        reinterpret_cast<const ElementAB**>(a_ptrs.data_ptr()),
        reinterpret_cast<StrideA*>(strides_a.data_ptr()),
        reinterpret_cast<const ElementAB**>(b_ptrs.data_ptr()),
        reinterpret_cast<StrideB*>(strides_b.data_ptr()),
        reinterpret_cast<const ElementBlockScale**>(sfa_ptrs.data_ptr()),
        reinterpret_cast<LayoutSFA*>(layouts_sfa_dev.data_ptr()),
        reinterpret_cast<const ElementBlockScale**>(sfb_ptrs.data_ptr()),
        reinterpret_cast<LayoutSFB*>(layouts_sfb_dev.data_ptr())
    };

    // Epilogue args — simple LinearCombination (alpha=1, beta=0)
    typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
        {},
        nullptr,
        reinterpret_cast<StrideC*>(strides_c.data_ptr()),
        reinterpret_cast<ElementD**>(out_ptrs.data_ptr()),
        reinterpret_cast<StrideD*>(strides_c.data_ptr())
    };
    epilogue_args.thread.alpha = 1.0f;
    epilogue_args.thread.beta = 0.0f;

    // Hardware info
    int device_id = activations.device().index();
    cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)
    };

    typename Gemm::GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        prob_shape, mainloop_args, epilogue_args, hw_info
    };

    Gemm gemm_op;
    auto can_impl = gemm_op.can_implement(args);
    TORCH_CHECK(can_impl == cutlass::Status::kSuccess,
                "CUTLASS SM100 blockwise cannot implement: status=", (int)can_impl);

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size},
                                   torch::dtype(torch::kByte).device(dev));

    auto status = gemm_op.initialize(args, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM100 blockwise init failed: status=", (int)status);

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM100 blockwise run failed: status=", (int)status);

#else
    TORCH_CHECK(false, "CUTLASS SM100 not supported");
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// TF32 Grouped GEMM entry point for GEMM2
// A = [total_tokens, K] FP32 (SwiGLU output)
// B = [E, N, K] FP32 (pre-dequanted weights)
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void setup_tf32_pointers(
    int64_t* expert_offsets, int num_experts,
    float** a_ptrs, float** b_ptrs, float** out_ptrs,
    float* a_base, float* b_base, float* out_base,
    int64_t N, int64_t K)
{
    int e = threadIdx.x;
    if (e >= num_experts) return;
    int64_t off = expert_offsets[e];
    a_ptrs[e] = a_base + off * K;
    b_ptrs[e] = b_base + (int64_t)e * N * K;
    out_ptrs[e] = out_base + off * N;
}

void cutlass_moe_gemm_tf32(
    torch::Tensor& output,              // [total_tokens, N] fp32
    torch::Tensor const& activations,   // [total_tokens, K] fp32
    torch::Tensor const& weights,       // [E, N, K] fp32 (pre-dequanted)
    torch::Tensor const& expert_offsets, // [E] int64
    torch::Tensor const& problem_sizes, // [E, 3] int32
    int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto stream = at::cuda::getCurrentCUDAStream(activations.device().index());
    auto dev = activations.device();
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    int64_t N = output.size(1);
    int64_t K = activations.size(1);

    // Pointer arrays
    auto a_ptrs = torch::empty(num_experts, opts_i64);
    auto b_ptrs = torch::empty(num_experts, opts_i64);
    auto out_ptrs = torch::empty(num_experts, opts_i64);

    setup_tf32_pointers<<<1, num_experts, 0, stream>>>(
        static_cast<int64_t*>(expert_offsets.data_ptr()), num_experts,
        reinterpret_cast<float**>(a_ptrs.data_ptr()),
        reinterpret_cast<float**>(b_ptrs.data_ptr()),
        reinterpret_cast<float**>(out_ptrs.data_ptr()),
        static_cast<float*>(activations.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        N, K);

    // Strides
    auto strides_a = torch::full({num_experts}, K, opts_i64);
    auto strides_b = torch::full({num_experts}, K, opts_i64);
    auto strides_c = torch::full({num_experts}, N, opts_i64);

    // Problem shape
    ProblemShape::UnderlyingProblemShape* prob_shapes =
        static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());
    ProblemShape prob_shape{num_experts, prob_shapes, nullptr};

    typename TF32_Gemm::GemmKernel::MainloopArguments mainloop_args{
        reinterpret_cast<const float**>(a_ptrs.data_ptr()),
        reinterpret_cast<TF32_StrideA*>(strides_a.data_ptr()),
        reinterpret_cast<const float**>(b_ptrs.data_ptr()),
        reinterpret_cast<TF32_StrideB*>(strides_b.data_ptr())
    };

    typename TF32_Gemm::GemmKernel::EpilogueArguments epilogue_args{
        {},
        nullptr,
        reinterpret_cast<TF32_StrideC*>(strides_c.data_ptr()),
        reinterpret_cast<float**>(out_ptrs.data_ptr()),
        reinterpret_cast<TF32_StrideD*>(strides_c.data_ptr())
    };
    epilogue_args.thread.alpha = 1.0f;
    epilogue_args.thread.beta = 0.0f;

    int device_id = activations.device().index();
    cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)
    };

    typename TF32_Gemm::GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        prob_shape, mainloop_args, epilogue_args, hw_info
    };

    TF32_Gemm gemm_op;
    TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
                "CUTLASS SM100 TF32 grouped GEMM cannot implement");

    size_t workspace_size = TF32_Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size},
                                   torch::dtype(torch::kByte).device(dev));

    auto status = gemm_op.initialize(args, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS SM100 TF32 init failed");

    // SM100 kernels need large shared memory opt-in
    {
        int smem = TF32_GemmKernel::SharedStorageSize;
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(cutlass::device_kernel<TF32_GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    status = gemm_op.run(stream, nullptr, true);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS SM100 TF32 run failed");

#else
    TORCH_CHECK(false, "CUTLASS SM100 not supported");
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// BF16 Grouped GEMM
// A = [total_tokens, K] BF16, B = [E, N, K] BF16 (pre-dequanted)
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void setup_bf16_pointers(
    int64_t* expert_offsets, int num_experts,
    cutlass::bfloat16_t** a_ptrs, cutlass::bfloat16_t** b_ptrs, float** out_ptrs,
    cutlass::bfloat16_t* a_base, cutlass::bfloat16_t* b_base, float* out_base,
    int64_t N, int64_t K) {
    int e = threadIdx.x; if (e >= num_experts) return;
    int64_t off = expert_offsets[e];
    a_ptrs[e] = a_base + off * K;
    b_ptrs[e] = b_base + (int64_t)e * N * K;
    out_ptrs[e] = out_base + off * N;
}

void cutlass_moe_gemm_bf16(
    torch::Tensor& output,
    torch::Tensor const& activations,   // BF16
    torch::Tensor const& weights,       // BF16
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto stream = at::cuda::getCurrentCUDAStream(activations.device().index());
    auto dev = activations.device();
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    int64_t N = output.size(1);
    int64_t K = activations.size(1);

    auto a_ptrs = torch::empty(num_experts, opts_i64);
    auto b_ptrs = torch::empty(num_experts, opts_i64);
    auto out_ptrs = torch::empty(num_experts, opts_i64);

    setup_bf16_pointers<<<1, num_experts, 0, stream>>>(
        static_cast<int64_t*>(expert_offsets.data_ptr()), num_experts,
        reinterpret_cast<cutlass::bfloat16_t**>(a_ptrs.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t**>(b_ptrs.data_ptr()),
        reinterpret_cast<float**>(out_ptrs.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(activations.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(weights.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        N, K);

    auto strides_a = torch::full({num_experts}, K, opts_i64);
    auto strides_b = torch::full({num_experts}, K, opts_i64);
    auto strides_c = torch::full({num_experts}, N, opts_i64);

    ProblemShape::UnderlyingProblemShape* prob_shapes =
        static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());
    ProblemShape prob_shape{num_experts, prob_shapes, nullptr};

    typename BF16_Gemm::GemmKernel::MainloopArguments mainloop_args{
        reinterpret_cast<const cutlass::bfloat16_t**>(a_ptrs.data_ptr()),
        reinterpret_cast<BF16_StrideA*>(strides_a.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t**>(b_ptrs.data_ptr()),
        reinterpret_cast<BF16_StrideB*>(strides_b.data_ptr())
    };

    typename BF16_Gemm::GemmKernel::EpilogueArguments epilogue_args{
        {}, nullptr,
        reinterpret_cast<BF16_StrideC*>(strides_c.data_ptr()),
        reinterpret_cast<float**>(out_ptrs.data_ptr()),
        reinterpret_cast<BF16_StrideD*>(strides_c.data_ptr())
    };
    epilogue_args.thread.alpha = 1.0f;
    epilogue_args.thread.beta = 0.0f;

    int device_id = activations.device().index();
    cutlass::KernelHardwareInfo hw_info{device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)};

    typename BF16_Gemm::GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        prob_shape, mainloop_args, epilogue_args, hw_info};

    BF16_Gemm gemm_op;
    TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
                "CUTLASS SM100 BF16 grouped GEMM cannot implement");

    size_t ws = BF16_Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, torch::dtype(torch::kByte).device(dev));
    auto status = gemm_op.initialize(args, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS SM100 BF16 init failed");
    status = gemm_op.run(stream, nullptr, true);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS SM100 BF16 run failed");

#else
    TORCH_CHECK(false, "CUTLASS SM100 not supported");
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flashinfer-style FP8 Blockwise Grouped GEMM — entry point
// A = [cum_m, K] fp8,  B = [num_groups, N, K] fp8
// SFA = [K//128, cum_m] fp32 (MN-major, per-token)
// SFB = [num_groups, K//128, N//128] fp32 (MN-major, per-128-block)
// m_indptr = [num_groups+1] int32 (cumulative, each entry multiple of 4)
// ═══════════════════════════════════════════════════════════════════════════════

void cutlass_moe_gemm_fp8_blockwise(
    torch::Tensor& output,          // [cum_m, N] fp32
    torch::Tensor const& A,         // [cum_m, K] fp8
    torch::Tensor const& B,         // [num_groups, N, K] fp8
    torch::Tensor const& SFA,       // [K//128, cum_m] fp32
    torch::Tensor const& SFB,       // [num_groups, K//128, N//128] fp32
    torch::Tensor const& m_indptr,  // [num_groups+1] int32
    int num_groups
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto stream = at::cuda::getCurrentCUDAStream(A.device().index());
    auto dev = A.device();
    int cum_m = A.size(0);
    int n = B.size(1);
    int k = B.size(2);
    int max_m = cum_m;

    auto opts_byte = torch::dtype(torch::kByte).device(dev);

    // Allocate device arrays for CUTLASS grouped GEMM arguments
    auto d_problem_sizes = torch::empty(
        {(int64_t)(num_groups * sizeof(ProblemShape::UnderlyingProblemShape))}, opts_byte);
    auto d_a_ptrs   = torch::empty({(int64_t)(num_groups * sizeof(void*))}, opts_byte);
    auto d_b_ptrs   = torch::empty({(int64_t)(num_groups * sizeof(void*))}, opts_byte);
    auto d_d_ptrs   = torch::empty({(int64_t)(num_groups * sizeof(void*))}, opts_byte);
    auto d_sfa_ptrs = torch::empty({(int64_t)(num_groups * sizeof(void*))}, opts_byte);
    auto d_sfb_ptrs = torch::empty({(int64_t)(num_groups * sizeof(void*))}, opts_byte);
    auto d_stride_a   = torch::empty({(int64_t)(num_groups * sizeof(fi_gemm::FI_StrideA))}, opts_byte);
    auto d_stride_b   = torch::empty({(int64_t)(num_groups * sizeof(fi_gemm::FI_StrideB))}, opts_byte);
    auto d_stride_d   = torch::empty({(int64_t)(num_groups * sizeof(fi_gemm::FI_StrideD))}, opts_byte);
    auto d_layout_sfa = torch::empty({(int64_t)(num_groups * sizeof(fi_gemm::LayoutSFA))}, opts_byte);
    auto d_layout_sfb = torch::empty({(int64_t)(num_groups * sizeof(fi_gemm::LayoutSFB))}, opts_byte);

    // Launch pointer-setup kernel with PDL (programmatic dependent launch)
    {
        int threads = std::min(num_groups, 1024);
        int blocks  = (num_groups + threads - 1) / threads;

        cudaLaunchConfig_t config = {};
        config.gridDim     = blocks;
        config.blockDim    = threads;
        config.dynamicSmemBytes = 0;
        config.stream      = stream;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = true;
        config.numAttrs = 1;
        config.attrs    = attrs;

        CUDA_CHECK(cudaLaunchKernelEx(
            &config, setup_fi_gemm_args,
            reinterpret_cast<ElementAB*>(A.data_ptr()),
            reinterpret_cast<ElementAB*>(B.data_ptr()),
            static_cast<float*>(SFA.data_ptr()),
            static_cast<float*>(SFB.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<int*>(m_indptr.data_ptr()),
            max_m, n, k, num_groups,
            reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(d_problem_sizes.data_ptr()),
            reinterpret_cast<const ElementAB**>(d_a_ptrs.data_ptr()),
            reinterpret_cast<const ElementAB**>(d_b_ptrs.data_ptr()),
            reinterpret_cast<const float**>(d_sfa_ptrs.data_ptr()),
            reinterpret_cast<const float**>(d_sfb_ptrs.data_ptr()),
            reinterpret_cast<float**>(d_d_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideA*>(d_stride_a.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideB*>(d_stride_b.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideD*>(d_stride_d.data_ptr()),
            reinterpret_cast<fi_gemm::LayoutSFA*>(d_layout_sfa.data_ptr()),
            reinterpret_cast<fi_gemm::LayoutSFB*>(d_layout_sfb.data_ptr())
        ));
    }

    // Hardware info
    int device_id = dev.index();
    cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)
    };

    // CUTLASS arguments — problem_sizes on device, launched with PDL
    typename fi_gemm::FI_Gemm::GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups,
         reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(d_problem_sizes.data_ptr()),
         /*host=*/nullptr},
        {   // mainloop
            reinterpret_cast<const ElementAB**>(d_a_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideA*>(d_stride_a.data_ptr()),
            reinterpret_cast<const ElementAB**>(d_b_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideB*>(d_stride_b.data_ptr()),
            reinterpret_cast<const float**>(d_sfa_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::LayoutSFA*>(d_layout_sfa.data_ptr()),
            reinterpret_cast<const float**>(d_sfb_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::LayoutSFB*>(d_layout_sfb.data_ptr()),
        },
        {   // epilogue
            {},       // thread (alpha/beta set below)
            nullptr,  // C_ptr
            nullptr,  // stride_C
            reinterpret_cast<float**>(d_d_ptrs.data_ptr()),
            reinterpret_cast<fi_gemm::FI_StrideD*>(d_stride_d.data_ptr()),
        },
        hw_info
    };
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta  = 0.0f;

    fi_gemm::FI_Gemm gemm_op;
    auto can_impl = gemm_op.can_implement(args);
    TORCH_CHECK(can_impl == cutlass::Status::kSuccess,
                "CUTLASS SM100 FP8 blockwise cannot implement: status=", (int)can_impl);

    size_t workspace_size = fi_gemm::FI_Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size}, opts_byte);

    auto status = gemm_op.initialize(args, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM100 FP8 blockwise init failed: status=", (int)status);

    status = gemm_op.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM100 FP8 blockwise run failed: status=", (int)status);

#else
    TORCH_CHECK(false, "CUTLASS SM100 not supported — compile with -arch=sm_100a");
#endif
}
