"""
CUTLASS MoE kernel v3 — experimental variant of moe_cutlass_v2.

Definition : moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Language   : python  (JIT-compiles CUDA/CUTLASS extension at runtime)
DPS        : true  (output pre-allocated bfloat16 [T, H], passed as last arg)
Entry      : kernel.py::kernel

moe_cutlass_v2 is kept frozen. Experimental changes go here (moe_cutlass_v3).
"""

import os
import pathlib
import torch
import torch.utils.cpp_extension as cpp_ext

_mod = None  # Cached compiled module


def _get_cutlass_include():
    """Find CUTLASS headers. Checks env var, common paths, then flashinfer's bundled copy."""
    # 1. Explicit env var takes priority (eval environment can override)
    env_path = os.environ.get("CUTLASS_PATH", "")
    if env_path and pathlib.Path(env_path).is_dir():
        return env_path

    # 2. Common system locations
    for candidate in [
        "/usr/local/cutlass/include",
        "/usr/include/cutlass",
    ]:
        if (pathlib.Path(candidate) / "cutlass" / "cutlass.h").exists():
            return candidate

    # 3. flashinfer's bundled copy (last resort)
    try:
        import flashinfer
        p = pathlib.Path(flashinfer.__file__).parent / "data" / "cutlass" / "include"
        if p.exists():
            return str(p)
    except ImportError:
        pass

    raise RuntimeError("CUTLASS headers not found. Set CUTLASS_PATH or install flashinfer.")


def _detect_cuda_arch():
    """Detect GPU compute capability and return gencode flag."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    cap = torch.cuda.get_device_capability(0)
    arch = cap[0] * 10 + cap[1]  # e.g., 90, 100
    # Use 'a' suffix for SM90a and SM100a (full feature set)
    return f"-gencode=arch=compute_{arch}a,code=sm_{arch}a"


def _build_extension():
    """JIT-compile the CUDA/CUTLASS MoE extension."""
    global _mod
    if _mod is not None:
        return _mod

    # Source files live alongside kernel.py. flashinfer-bench's
    # pack_solution_from_files doesn't recurse, so nested csrc/ subdirs would
    # get dropped when the solution is packed/replayed in worker processes.
    src_dir = pathlib.Path(__file__).parent
    sources = [
        str(src_dir / "main.cpp"),
        str(src_dir / "kernel.cu"),
        str(src_dir / "moe_grouped_gemm_fp8.cu"),
    ]

    for s in sources:
        if not pathlib.Path(s).exists():
            raise RuntimeError(f"Source file not found: {s}")

    cutlass_include = _get_cutlass_include()
    gencode = _detect_cuda_arch()

    _mod = cpp_ext.load(
        name="moe_cutlass_v3_ext",
        sources=sources,
        extra_cuda_cflags=[
            gencode,
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-O2",
        ],
        extra_include_paths=[str(src_dir), cutlass_include],
        verbose=True,
    )
    return _mod


def kernel(
    routing_logits,        # [T, 256]         float32
    routing_bias,          # [256]             bfloat16
    hidden_states,         # [T, 7168]         fp8_e4m3fn
    hidden_states_scale,   # [56, T]           float32
    gemm1_weights,         # [32, 4096, 7168]  fp8_e4m3fn
    gemm1_weights_scale,   # [32, 32, 56]      float32
    gemm2_weights,         # [32, 7168, 2048]  fp8_e4m3fn
    gemm2_weights_scale,   # [32, 56, 16]      float32
    local_expert_offset,   # int
    routed_scaling_factor, # float
    output,                # [T, 7168]         bfloat16 (DPS, pre-allocated)
):
    """MoE forward pass using CUTLASS SM100a grouped GEMM."""
    mod = _build_extension()
    # C++ writes directly into the pre-allocated DPS output tensor,
    # avoiding an intermediate bf16 allocation + Python-level copy.
    mod.run(
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        int(local_expert_offset),
        float(routed_scaling_factor),
        output,
    )
