"""
Triton MoE kernel — FP8 block-scale DeepSeek-V3 routing, grouped GEMM.

Definition : moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Language   : triton
DPS        : true  (output pre-allocated bfloat16 [T, H], passed as last arg)
Entry      : kernel.py::kernel

Key optimizations:
  1. Fused routing kernel.
  2. Token permutation — sort tokens by expert, pad to BLOCK_M for grouped GEMM.
  3. Grouped FP8 GEMM1 — one kernel for all 32 experts, in-kernel token gather.
  4. Fused SwiGLU kernel — activation in one pass, no intermediate materialization.
  5. Grouped GEMM2 — fp32 tl.dot (TF32), routing weight fused in epilogue.
  6. SPLIT_K for small batches to increase GPU parallelism.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_H = 7168            # hidden size
_I = 2048            # intermediate size
_BLOCK_SCALE = 128   # FP8 quantization block size
_E_GLOBAL = 256      # total experts
_TOP_K = 8           # experts per token
_N_GROUP = 8         # routing groups
_TOPK_GROUP = 4      # groups to keep
_GROUP_SIZE = 32     # experts per group (256 / 8)
_GEMM_BLOCK_M = 128  # token-dimension padding for grouped GEMM


# ---------------------------------------------------------------------------
# Kernel 1 — Fused DeepSeek-V3 no-aux routing
#   Outputs: topk_ids [T, 8] int32, topk_weights [T, 8] float32.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_routing_kernel(
    logits_ptr,              # [T, 256] float32
    bias_ptr,                # [256]    bfloat16
    topk_ids_ptr,            # [T, 8]   int32   (output)
    topk_weights_ptr,        # [T, 8]   float32 (output)
    routed_scaling_factor,   # float scalar
    T: tl.constexpr,
    E_GLOBAL: tl.constexpr,
    TOP_K: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    offs_e = tl.arange(0, E_GLOBAL)
    logits = tl.load(logits_ptr + pid * E_GLOBAL + offs_e).to(tl.float32)
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)

    s = tl.sigmoid(logits)
    s_with_bias = s + bias
    group_id = offs_e // GROUP_SIZE

    # Group scoring: sum of top-2 per group
    NEG_INF: tl.constexpr = float("-inf")
    g_offs = tl.arange(0, N_GROUP)
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)

    for g in tl.static_range(N_GROUP):
        in_group = (group_id == g)
        masked = tl.where(in_group, s_with_bias, NEG_INF)
        idx1 = tl.argmax(masked, axis=0)
        max1 = tl.max(masked, axis=0)
        masked2 = tl.where(offs_e == idx1, NEG_INF, masked)
        max2 = tl.max(masked2, axis=0)
        max2 = tl.where(max2 == NEG_INF, 0.0, max2)
        group_scores = tl.where(g_offs == g, max1 + max2, group_scores)

    # Top-4 group selection
    g_scores_tmp = group_scores
    group_mask = tl.zeros([N_GROUP], dtype=tl.int32)
    for _ in tl.static_range(TOPK_GROUP):
        best_g = tl.argmax(g_scores_tmp, axis=0)
        group_mask = tl.where(g_offs == best_g, 1, group_mask)
        g_scores_tmp = tl.where(g_offs == best_g, NEG_INF, g_scores_tmp)

    # Expert mask from group selection
    expert_selected = tl.zeros([E_GLOBAL], dtype=tl.int32)
    for g in tl.static_range(N_GROUP):
        in_group = (group_id == g)
        g_sel = tl.sum(tl.where(g_offs == g, group_mask, 0))
        expert_selected = tl.where(in_group & (g_sel > 0), 1, expert_selected)

    scores_pruned = tl.where(expert_selected == 1, s_with_bias, NEG_INF)

    # Top-K expert selection
    topk_ids = tl.zeros([TOP_K], dtype=tl.int32)
    topk_s = tl.zeros([TOP_K], dtype=tl.float32)
    k_offs = tl.arange(0, TOP_K)

    for k in tl.static_range(TOP_K):
        best_e = tl.argmax(scores_pruned, axis=0)
        topk_ids = tl.where(k_offs == k, best_e, topk_ids)
        best_s = tl.sum(tl.where(offs_e == best_e, s, 0.0))
        topk_s = tl.where(k_offs == k, best_s, topk_s)
        scores_pruned = tl.where(offs_e == best_e, NEG_INF, scores_pruned)

    # Normalize weights
    weight_sum = tl.sum(topk_s) + 1e-20
    topk_weights = (topk_s / weight_sum) * routed_scaling_factor

    tl.store(topk_ids_ptr + pid * TOP_K + k_offs, topk_ids)
    tl.store(topk_weights_ptr + pid * TOP_K + k_offs, topk_weights)


# ---------------------------------------------------------------------------
# Kernel 2 — Fused SwiGLU
#   G1 = [X1 | X2] with shape [Tk, 2*I].
#   Output C = silu(X2) * X1 with shape [Tk, I].
# ---------------------------------------------------------------------------

@triton.jit
def _swiglue_kernel(
    G1_ptr, C_ptr, Tk,
    I: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_t[:, None] < Tk) & (offs_i[None, :] < I)

    x1 = tl.load(
        G1_ptr + offs_t[:, None] * (2 * I) + offs_i[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    x2 = tl.load(
        G1_ptr + offs_t[:, None] * (2 * I) + (I + offs_i[None, :]),
        mask=mask, other=0.0,
    ).to(tl.float32)

    c = (x2 * tl.sigmoid(x2)) * x1
    tl.store(C_ptr + offs_t[:, None] * I + offs_i[None, :], c, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3 — Grouped GEMM (all experts in one launch)
#   Modes (via constexpr flags):
#     GEMM1: GATHER_A=True, USE_FP8_A=True, MUL_ROUTED_WEIGHT=False
#     GEMM2: GATHER_A=False, USE_FP8_A=False, MUL_ROUTED_WEIGHT=True
#   SPLIT_K for small batches (T<=80).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_SIZE_M": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _grouped_gemm_kernel(
    A_ptr, A_scale_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, position_weights_ptr,
    T, N, K, total_padded,
    stride_at, stride_ak,
    stride_ask, stride_asm,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    GATHER_A: tl.constexpr,
    USE_FP8_A: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_k_id = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = total_padded // BLOCK_M

    # L2 cache swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert_id = tl.load(expert_ids_ptr + pid_m * BLOCK_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tok_ids = tl.load(sorted_token_ids_ptr + offs_m)
    token_mask = tok_ids < T
    tok_ids_safe = tl.where(token_mask, tok_ids, 0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_base = split_k_id * k_per_split
    k_end = min(k_start_base + k_per_split, K)

    for k_start in range(k_start_base, k_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_block = k_start // BLOCK_K

        # ── Load A tile [BLOCK_M, BLOCK_K] ──
        if GATHER_A:
            a_ptrs = A_ptr + tok_ids_safe[:, None] * stride_at + offs_k[None, :] * stride_ak
        else:
            a_ptrs = A_ptr + offs_m[:, None] * stride_at + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K), other=0.0)

        # ── Load B tile [BLOCK_K, BLOCK_N] ──
        b_ptrs = (B_ptr + expert_id * stride_be
                  + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # ── B scale (always needed) ──
        n_block_idx = offs_n // BLOCK_K
        b_s_ptrs = (B_scale_ptr + expert_id * stride_bse
                    + n_block_idx * stride_bsn + k_block * stride_bsk)
        b_scale = tl.load(b_s_ptrs, mask=offs_n < N, other=1.0)

        if USE_FP8_A:
            # FP8 tensor core path (GEMM1)
            partial = tl.dot(a, b)
            if GATHER_A:
                a_s_ptrs = A_scale_ptr + k_block * stride_ask + tok_ids_safe * stride_asm
            else:
                a_s_ptrs = A_scale_ptr + k_block * stride_ask + offs_m * stride_asm
            a_scale = tl.load(a_s_ptrs, mask=token_mask, other=0.0)
            acc += partial * (a_scale[:, None] * b_scale[None, :])
        else:
            # FP32/TF32 path (GEMM2)
            a_f32 = a.to(tl.float32)
            b_f32 = b.to(tl.float32)
            partial = tl.dot(a_f32, b_f32)
            acc += partial * b_scale[None, :]

    # ── Epilogue: routing weight ──
    if MUL_ROUTED_WEIGHT:
        w = tl.load(position_weights_ptr + offs_m, mask=token_mask, other=0.0)
        acc *= w[:, None]

    # ── Store ──
    if SPLIT_K > 1:
        c_ptrs = (C_ptr + split_k_id * total_padded * N
                  + offs_m[:, None] * N + offs_n[None, :])
    else:
        c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _run_routing(routing_logits, routing_bias, routed_scaling_factor, T):
    """Launch fused routing → topk_ids [T,8] int32, topk_weights [T,8] fp32."""
    device = routing_logits.device
    topk_ids = torch.empty(T, _TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(T, _TOP_K, dtype=torch.float32, device=device)
    _fused_routing_kernel[(T,)](
        routing_logits, routing_bias, topk_ids, topk_weights,
        float(routed_scaling_factor),
        T=T, E_GLOBAL=_E_GLOBAL, TOP_K=_TOP_K,
        N_GROUP=_N_GROUP, TOPK_GROUP=_TOPK_GROUP, GROUP_SIZE=_GROUP_SIZE,
    )
    return topk_ids, topk_weights


def _moe_align_block_size(topk_ids, topk_weights, T, E_local, local_start):
    """Sort tokens by expert, pad to BLOCK_M, precompute per-position weights."""
    device = topk_ids.device
    BLOCK_M = _GEMM_BLOCK_M

    token_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, _TOP_K).reshape(-1)
    expert_indices = topk_ids.reshape(-1).to(torch.int64)
    weight_values = topk_weights.reshape(-1)

    local_end = local_start + E_local
    local_mask = (expert_indices >= local_start) & (expert_indices < local_end)
    local_tokens = token_indices[local_mask]
    local_experts = (expert_indices[local_mask] - local_start).to(torch.int64)
    local_weights = weight_values[local_mask]

    if local_tokens.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
            0,
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    sort_idx = torch.argsort(local_experts, stable=True)
    sorted_tokens = local_tokens[sort_idx]
    sorted_experts = local_experts[sort_idx]
    sorted_weights = local_weights[sort_idx]

    expert_counts = torch.zeros(E_local, dtype=torch.int64, device=device)
    expert_counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts))
    padded_counts = ((expert_counts + BLOCK_M - 1) // BLOCK_M) * BLOCK_M

    total_padded = padded_counts.sum().item()
    out_tokens = torch.full((total_padded,), T, dtype=torch.int64, device=device)
    out_weights = torch.zeros(total_padded, dtype=torch.float32, device=device)
    out_expert_ids = torch.zeros(total_padded, dtype=torch.int32, device=device)

    src_offset = 0
    dst_offset = 0
    for e in range(E_local):
        n = expert_counts[e].item()
        pn = padded_counts[e].item()
        if pn == 0:
            continue
        out_tokens[dst_offset:dst_offset + n] = sorted_tokens[src_offset:src_offset + n]
        out_weights[dst_offset:dst_offset + n] = sorted_weights[src_offset:src_offset + n]
        out_expert_ids[dst_offset:dst_offset + pn] = e
        src_offset += n
        dst_offset += pn

    return out_tokens, out_expert_ids, total_padded, out_weights


def _launch_grouped_gemm(
    A, A_scale, B, B_scale,
    sorted_token_ids, expert_ids, position_weights,
    T, total_padded, N, K,
    gather_a, use_fp8_a, mul_routed_weight,
):
    """Launch grouped GEMM with automatic SPLIT_K for small batches."""
    device = A.device

    # SPLIT_K heuristic: more splits for smaller batches
    if total_padded <= 64:
        split_k = 8
    elif total_padded <= 256:
        split_k = 4
    elif total_padded <= 1024:
        split_k = 2
    else:
        split_k = 1

    fuse_weight = mul_routed_weight and (split_k == 1)

    if A_scale is None:
        A_scale = torch.ones(1, 1, dtype=torch.float32, device=device)

    if split_k > 1:
        C_buf = torch.zeros(split_k, total_padded, N, dtype=torch.float32, device=device)
    else:
        C_buf = torch.empty(total_padded, N, dtype=torch.float32, device=device)

    grid = lambda META: (
        triton.cdiv(total_padded, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )

    _grouped_gemm_kernel[grid](
        A, A_scale, B, B_scale, C_buf,
        sorted_token_ids, expert_ids, position_weights,
        T, N, K, total_padded,
        A.stride(0), A.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        B_scale.stride(0), B_scale.stride(1), B_scale.stride(2),
        GATHER_A=gather_a,
        USE_FP8_A=use_fp8_a,
        MUL_ROUTED_WEIGHT=fuse_weight,
        SPLIT_K=split_k,
        BLOCK_K=_BLOCK_SCALE,
    )

    if split_k > 1:
        C = C_buf.sum(dim=0)
        if mul_routed_weight:
            C *= position_weights.unsqueeze(1)
    else:
        C = C_buf

    return C


# ---------------------------------------------------------------------------
# Main entry point (DPS: output is the last parameter)
# ---------------------------------------------------------------------------

@torch.no_grad()
def kernel(
    routing_logits,        # [T, 256]       float32
    routing_bias,          # [256]           bfloat16
    hidden_states,         # [T, 7168]      fp8_e4m3fn
    hidden_states_scale,   # [56, T]        float32
    gemm1_weights,         # [32, 4096, 7168] fp8_e4m3fn
    gemm1_weights_scale,   # [32, 32, 56]   float32
    gemm2_weights,         # [32, 7168, 2048] fp8_e4m3fn
    gemm2_weights_scale,   # [32, 56, 16]   float32
    local_expert_offset,   # int
    routed_scaling_factor, # float
    output,                # [T, 7168]      bfloat16 (DPS, pre-allocated)
):
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    # ── 1. Fused routing (1 kernel launch) ────────────────────────────────────
    topk_ids, topk_weights = _run_routing(
        routing_logits, routing_bias, routed_scaling_factor, T,
    )

    # ── 2. Token permutation + BLOCK_M padding ───────────────────────────────
    sorted_token_ids, expert_ids, total_padded, position_weights = (
        _moe_align_block_size(topk_ids, topk_weights, T, E_local, local_start)
    )

    if total_padded == 0:
        output.zero_()
        return

    # ── 3. Grouped FP8 GEMM1 (in-kernel gather, no pre-copy) ─────────────────
    G1 = _launch_grouped_gemm(
        hidden_states, hidden_states_scale.contiguous(),
        gemm1_weights, gemm1_weights_scale.contiguous(),
        sorted_token_ids, expert_ids, position_weights,
        T, total_padded, 2 * _I, _H,
        gather_a=True, use_fp8_a=True, mul_routed_weight=False,
    )

    # ── 4. Fused SwiGLU ──────────────────────────────────────────────────────
    C = torch.empty(total_padded, _I, dtype=torch.float32, device=device)
    grid_sw = (triton.cdiv(total_padded, 32), triton.cdiv(_I, 128))
    _swiglue_kernel[grid_sw](G1, C, total_padded, I=_I, BLOCK_T=32, BLOCK_I=128)

    # ── 5. Grouped GEMM2 (TF32, routing weight fused in epilogue) ────────────
    O = _launch_grouped_gemm(
        C, None,
        gemm2_weights, gemm2_weights_scale.contiguous(),
        sorted_token_ids, expert_ids, position_weights,
        T, total_padded, _H, _I,
        gather_a=False, use_fp8_a=False, mul_routed_weight=True,
    )

    # ── 6. Scatter-add to output ─────────────────────────────────────────────
    valid_mask = sorted_token_ids < T
    valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(1)
    valid_token_ids = sorted_token_ids[valid_pos]

    acc = torch.zeros(T, _H, dtype=torch.float32, device=device)
    acc.index_add_(0, valid_token_ids, O[valid_pos])

    output.copy_(acc.to(torch.bfloat16))
