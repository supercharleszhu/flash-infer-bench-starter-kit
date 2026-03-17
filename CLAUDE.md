# FlashInfer-Bench Starter Kit — Skills & Workflow

Quick reference synthesized from README.md, SOLUTION_NOTES.md, and COMPILATION_WORKFLOW.md.

---

## Project Overview

**Contest:** FlashInfer AI Kernel Generation Contest @ MLSys 2026
**Track:** `sparse_attention` — DSA (DeepSeek Native Sparse Attention)
**Target definition:** `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
**Current implementation:** Python reference (`solution/python/kernel.py`), `dps=false`

---

## Environment Setup

### Local (this machine)

```bash
# Python with flashinfer_bench installed
.venv/bin/python3    # at project root

# Dataset
export FIB_DATASET_PATH=/home/chzhu/mlsys26-contest
```

### Remote Pod (Kubernetes)

- **Pod:** `f488d21f8ceb742b49c6-n1-0-master-0`
- **Namespace:** `training-km`
- **Python with torch+flashinfer_bench:** `/usr/local/bin/python3` (Python 3.12, torch 2.10.0.1+cu130)
- **Dataset path on pod:** `/shared/public/sharing/chzhu/mlsys26-contest`
- **Project path on pod:** `/shared/public/sharing/chzhu/flash-infer-bench-starter-kit/`
- **Wheels:** `/shared/public/sharing/chzhu/fib-wheels/`

---

## Core Workflow

### 1. Pack Solution

```bash
# Local
.venv/bin/python3 scripts/pack_solution.py

# On pod
python3 scripts/pack_solution.py
```

Reads `config.toml` + `solution/python/kernel.py` → writes `solution.json` (with SHA1 hash).

### 2. Run Benchmark Locally (on pod)

```bash
# On pod — foreground
cd /shared/public/sharing/chzhu/flash-infer-bench-starter-kit
export FIB_DATASET_PATH=/shared/public/sharing/chzhu/mlsys26-contest
python3 scripts/run_local.py

# Background
nohup python3 scripts/run_local.py > benchmark_run.log 2>&1 &
```

### 3. Run via kubectl (from local machine)

```bash
# Start benchmark in background
kubectl exec -n training-km f488d21f8ceb742b49c6-n1-0-master-0 -- bash -c "
  cd /shared/public/sharing/chzhu/flash-infer-bench-starter-kit
  export FIB_DATASET_PATH=/shared/public/sharing/chzhu/mlsys26-contest
  nohup python3 scripts/run_local.py > benchmark_run.log 2>&1 &
  echo \"Started PID \$!\"
"

# Check progress
kubectl exec -n training-km f488d21f8ceb742b49c6-n1-0-master-0 -- \
  cat /shared/public/sharing/chzhu/flash-infer-bench-starter-kit/benchmark_run.log

# Tail live log
kubectl exec -n training-km f488d21f8ceb742b49c6-n1-0-master-0 -- \
  tail -f /shared/public/sharing/chzhu/flash-infer-bench-starter-kit/benchmark_run.log
```

### 4. Sync Files to Pod

```bash
# Sync updated solution files
kubectl cp solution/python/kernel.py \
  training-km/f488d21f8ceb742b49c6-n1-0-master-0:\
/shared/public/sharing/chzhu/flash-infer-bench-starter-kit/solution/python/kernel.py

kubectl cp solution.json \
  training-km/f488d21f8ceb742b49c6-n1-0-master-0:\
/shared/public/sharing/chzhu/flash-infer-bench-starter-kit/solution.json

kubectl cp config.toml \
  training-km/f488d21f8ceb742b49c6-n1-0-master-0:\
/shared/public/sharing/chzhu/flash-infer-bench-starter-kit/config.toml
```

---

## Installing Dependencies on a New Pod

flashinfer_bench must be installed into the Python that has torch:

```bash
kubectl exec -n training-km <pod> -- bash -c "
  WHEELS=/shared/public/sharing/chzhu/fib-wheels
  pip3 install --no-deps \
    \$WHEELS/flashinfer_bench-0.1.2-py3-none-any.whl \
    \$WHEELS/apache_tvm_ffi-0.1.9-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    \$WHEELS/docstring_parser-0.17.0-py3-none-any.whl \
    \$WHEELS/safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
"
```

Use `--no-deps` to avoid torch version conflict (pod has torch 2.10, wheel requires >=2.8).
Also install `pydantic` and `packaging` if missing:

```bash
kubectl exec -n training-km <pod> -- pip3 install pydantic packaging filelock
```

---

## Key Configuration (`config.toml`)

```toml
[solution]
name = "supercharleszhu-sparse-attention-v1"
definition = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
author = "supercharleszhu"

[build]
language = "python"
entry_point = "kernel.py::kernel"
destination_passing_style = false   # kernel returns (output, lse) — not DPS
```

---

## Compilation Pipeline

```
solution/python/kernel.py
    ↓ pack_solution_from_files()     # reads .py, computes SHA1 hash
Solution JSON (solution.json)
    ↓ BuilderRegistry.build()        # PythonBuilder: importlib.import_module()
Runnable (callable)
    ↓ PersistentRunner               # subprocess worker per GPU
Evaluation (correctness + perf)
    ↓ Benchmark.run_all()
TraceSet → ~/.cache/flashinfer_bench/traces/*.jsonl
```

Builder selection order: `TritonBuilder → TileLangBuilder → PythonBuilder → TVMFFIBuilder → TorchBuilder`

Build cache: `~/.cache/flashinfer_bench/` (keyed by solution SHA1 hash)

---

## Understanding the Kernel (sparse_attention)

### What the kernel computes

DeepSeek-V3.2 sparse attention: attend only to top-K=2048 pre-selected tokens (not full KV).

### MLA Split Keys

| Tensor | Shape | Role |
|--------|-------|------|
| `q_nope` | `[tokens, 16, 512]` | Content query (no positional encoding) |
| `q_pe` | `[tokens, 16, 64]` | Positional query (RoPE applied) |
| `ckv_cache` | `[pages, 64, 512]` | Compressed KV (doubles as Value — no separate V) |
| `kpe_cache` | `[pages, 64, 64]` | Key positional encoding (RoPE applied) |

```python
logits = (q_nope @ Kc.T) + (q_pe @ Kp.T)   # sm_scale = 1/sqrt(576)
attn   = softmax(logits * scale)
output = attn @ Kc                            # Kc reused as V
```

### Paged KV Cache

Flat index = `page_id * 64 + offset`:
```python
Kc_all = ckv_cache.reshape(-1, 512)   # [pages*64, 512]
Kc     = Kc_all[sparse_indices]       # gather top-K tokens
```

### Index Padding

`sparse_indices` shape `[tokens, 2048]`; invalid entries are `-1` (filter before gather).

---

## Benchmark Evaluation Logic

For each `(definition, workload, solution)`:

1. **Build** — load Python module, validate signature
2. **Baseline** — run reference impl, measure latency
3. **Correctness** — compare outputs: shape, dtype, abs/rel error
4. **Performance** — `speedup = reference_latency / solution_latency` (only if correct)

**Failure policy:** 3 consecutive failures → skip remaining workloads; `COMPILE_ERROR` → skip all immediately.

---

## Current Results (T4 local baseline)

2 of 23 workloads pass on T4 (16 GiB) — small workloads only. Larger workloads OOM on T4 but expected to pass on B200.

| Workload | Latency | Speedup | Abs Error |
|----------|---------|---------|-----------|
| `0c23b10c` | 7.447 ms | 1.01x | 0.00 |
| `b7668cfd` | 7.894 ms | 1.01x | 0.00 |

---

## Checking Benchmark Results

```bash
# View traces (JSONL)
kubectl exec -n training-km f488d21f8ceb742b49c6-n1-0-master-0 -- \
  cat ~/.cache/flashinfer_bench/traces/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl
```

---

## DPS Note

`destination_passing_style = false` means the kernel **returns** `(output, lse)` instead of writing to pre-allocated tensors. If you see:

```
Destination-passing style callable: expected XX parameters, but got XX
```

Check that `dps=false` is set in config.toml and solution.json spec.
