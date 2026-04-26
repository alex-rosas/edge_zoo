"""
benchmark/bench.py
------------------
Main benchmark script: runs all three models through the PTQ pipeline
and produces the three result tables for the engineering report.

Output files:
  results/tradeoff.csv    — accuracy delta, size, latency per model
  results/layer_errors.csv — layer-wise L2 and max error per model
  results/onnx_graph.csv  — quantized vs FP32 operator counts per model

This script answers the governing question of the project (proposal, Section 1):
  Given three CNN architectures with different accuracy and efficiency
  profiles, which should be deployed on an edge device with a 4 MB memory
  budget and a 10 ms latency constraint — and where does INT8 quantization
  cost most?

Latency measurement note:
  Latency is measured on CPU with torch.no_grad() over N_LATENCY_RUNS
  repetitions. The first run is excluded (warm-up). This is not a
  production profiler — it is sufficient to observe the relative
  FP32 vs INT8 scaling behaviour described in Hypothesis H2.
  Real NPU latency requires hardware deployment (out of scope,
  proposal Section 2.2).
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from models.zoo import get_zoo, print_zoo_summary
from quantize.pipeline import run_pipeline
from quantize.observers import real_calibration_loader, evaluate_top1
from quantize.error_analysis import run_error_attribution
from quantize.onnx_interrogate import run_onnx_interrogation


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR        = "./data"
OUTPUT_DIR      = "onnx_models"
RESULTS_DIR     = "results"
N_LATENCY_RUNS  = 50      # repetitions per latency measurement
N_CAL_BATCHES   = 100     # calibration batches (proposal, Section 4.2)
N_EVAL_BATCHES  = 50      # evaluation batches (~1600 images)
DEVICE          = "cpu"   # CPU for reproducibility; GPU optional


# ─── Latency measurement ──────────────────────────────────────────────────────

def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    n_runs: int = N_LATENCY_RUNS,
    device: str = DEVICE,
) -> float:
    """
    Measure mean inference latency in milliseconds over n_runs repetitions.

    The first forward pass is excluded as warm-up — it triggers
    kernel compilation and cache population that would inflate the
    first measurement. Subsequent runs are representative of steady-state
    inference latency.

    Why wall-clock time and not FLOP counts:
      FLOP counts predict arithmetic throughput but not real latency.
      Real latency includes memory access patterns, operator dispatch
      overhead, and requantization cost — all of which are relevant to
      the H2 prediction that latency improvement will be smaller than
      the size reduction factor.

    Args:
        model:       Any nn.Module (FP32 or INT8).
        input_shape: Tensor shape excluding batch dim, e.g. (3, 224, 224).
        n_runs:      Repetitions. Default: N_LATENCY_RUNS.
        device:      Compute device. Default: DEVICE.

    Returns:
        Mean latency in milliseconds (excluding warm-up).
    """
    model.eval()
    model.to(device)
    dummy = torch.zeros(1, *input_shape).to(device)

    with torch.no_grad():
        # Warm-up — excluded from timing.
        model(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)   # convert to ms

    return sum(times) / len(times)


# ─── Model size measurement ───────────────────────────────────────────────────

def measure_onnx_size_mb(onnx_path: str) -> float:
    """
    Return the actual file size of an exported ONNX model in megabytes.

    This is the real size — including graph metadata, operator attributes,
    and weight tensors — not the theoretical floor computed from parameter
    count in the zoo. The theoretical floor (n_params × bytes_per_element)
    is useful for pre-quantization planning; the actual file size is what
    matters for the 4 MB deployment constraint.

    Args:
        onnx_path: Path to the .onnx file.

    Returns:
        File size in MB.
    """
    size_bytes = Path(onnx_path).stat().st_size
    return size_bytes / (1024 ** 2)


# ─── Per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(entry, eval_loader) -> dict:
    """
    Run the full benchmark for one ModelEntry.

    Executes: pipeline → latency → size → accuracy → error attribution.
    Returns a flat dict of scalar results for the tradeoff table, plus
    the error DataFrame for layer_errors.csv.

    The FP32 latency and accuracy are measured on the original model
    from the zoo (untouched). The INT8 latency and accuracy are measured
    on the quantized model returned by run_pipeline().
    """
    name = entry.name
    print(f"\n{'═'*60}")
    print(f" Benchmarking: {name}")
    print(f"{'═'*60}")

    # ── PTQ pipeline ──────────────────────────────────────────────────────────
    cal_loader = real_calibration_loader(
        data_dir=DATA_DIR,
        batch_size=32,
        image_size=entry.input_shape[1],
    )

    pipeline_result = run_pipeline(
        entry,
        calibration_loader=cal_loader,
        output_dir=OUTPUT_DIR,
        n_cal_batches=N_CAL_BATCHES,
        device=DEVICE,
    )

    fp32_model = pipeline_result["fp32_model"]
    int8_model = pipeline_result["int8_model"]
    onnx_path  = pipeline_result["onnx_path"]

    # ── Latency ───────────────────────────────────────────────────────────────
    print(f"\n  Measuring FP32 latency ({N_LATENCY_RUNS} runs)...")
    fp32_latency = measure_latency(fp32_model, entry.input_shape)

    print(f"  Measuring INT8 latency ({N_LATENCY_RUNS} runs)...")
    int8_latency = measure_latency(int8_model, entry.input_shape)

    latency_ratio = fp32_latency / int8_latency if int8_latency > 0 else float("nan")

    # ── Model size ────────────────────────────────────────────────────────────
    fp32_size_mb = entry.size_fp32_mb    # theoretical floor from zoo metadata
    int8_size_mb = measure_onnx_size_mb(onnx_path)
    size_ratio   = fp32_size_mb / int8_size_mb if int8_size_mb > 0 else float("nan")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print("  Evaluating FP32 accuracy...")
    fp32_top1 = evaluate_top1(
        fp32_model, eval_loader, device=DEVICE, max_batches=N_EVAL_BATCHES
    )

    print("  Evaluating INT8 accuracy...")
    int8_top1 = evaluate_top1(
        int8_model, eval_loader, device=DEVICE, max_batches=N_EVAL_BATCHES
    )

    accuracy_delta = fp32_top1 - int8_top1

    # ── Error attribution ─────────────────────────────────────────────────────
    error_df = run_error_attribution(
        fp32_model, int8_model, entry, device=DEVICE
    )
    error_df["model"] = name   # tag rows for multi-model CSV

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_model_summary(
        name, fp32_top1, int8_top1, accuracy_delta,
        fp32_latency, int8_latency, latency_ratio,
        fp32_size_mb, int8_size_mb, size_ratio,
    )

    return {
        "scalar": {
            "model":           name,
            "arch_family":     entry.arch_family,
            "fp32_top1":       fp32_top1,
            "int8_top1":       int8_top1,
            "accuracy_delta":  accuracy_delta,
            "fp32_latency_ms": fp32_latency,
            "int8_latency_ms": int8_latency,
            "latency_ratio":   latency_ratio,
            "fp32_size_mb":    fp32_size_mb,
            "int8_size_mb":    int8_size_mb,
            "size_ratio":      size_ratio,
            "fits_4mb":        int8_size_mb <= 4.0,
            "onnx_path":       onnx_path,
        },
        "error_df": error_df,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

def _print_model_summary(
    name, fp32_top1, int8_top1, accuracy_delta,
    fp32_lat, int8_lat, lat_ratio,
    fp32_size, int8_size, size_ratio,
) -> None:
    print(f"\n  ── Summary: {name} ──")
    print(f"  {'Metric':<25} {'FP32':>10} {'INT8':>10} {'Ratio':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'Top-1 accuracy':<25} {fp32_top1*100:>9.2f}% {int8_top1*100:>9.2f}% {accuracy_delta*100:>+7.2f}%")
    print(f"  {'Latency (ms)':<25} {fp32_lat:>10.2f} {int8_lat:>10.2f} {lat_ratio:>7.2f}×")
    print(f"  {'Size (MB)':<25} {fp32_size:>10.2f} {int8_size:>10.2f} {size_ratio:>7.2f}×")
    fits = "✓ fits" if int8_size <= 4.0 else "✗ exceeds"
    print(f"  4 MB constraint: {fits} ({int8_size:.2f} MB)")


def _print_final_tradeoff(df: pd.DataFrame) -> None:
    """Print the cross-model tradeoff table — the governing question answer."""
    print(f"\n{'═'*70}")
    print(" TRADEOFF TABLE — governing question answer")
    print(f"{'═'*70}")
    print(f"  {'Model':<20} {'Δ Acc':>7} {'INT8 ms':>9} {'INT8 MB':>9} {'4MB?':>6} {'Lat×':>6}")
    print(f"  {'─'*20} {'─'*7} {'─'*9} {'─'*9} {'─'*6} {'─'*6}")

    for _, row in df.iterrows():
        fits = "✓" if row["fits_4mb"] else "✗"
        print(
            f"  {row['model']:<20} "
            f"{row['accuracy_delta']*100:>+6.2f}% "
            f"{row['int8_latency_ms']:>9.2f} "
            f"{row['int8_size_mb']:>9.2f} "
            f"{fits:>6} "
            f"{row['latency_ratio']:>5.2f}×"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n══ EdgeZoo Benchmark ══")
    print_zoo_summary()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    zoo = get_zoo()

    # Shared eval loader — same images across all models for comparability.
    torch.manual_seed(42)
    eval_loader = real_calibration_loader(
        data_dir=DATA_DIR,
        batch_size=32,
        image_size=224,
    )

    tradeoff_rows = []
    error_dfs     = []
    onnx_paths    = {}

    for entry in zoo.values():
        result = benchmark_model(entry, eval_loader)

        tradeoff_rows.append(result["scalar"])
        error_dfs.append(result["error_df"])
        onnx_paths[entry.name] = result["scalar"]["onnx_path"]

    # ── Assemble result tables ────────────────────────────────────────────────
    tradeoff_df  = pd.DataFrame(tradeoff_rows)
    layer_err_df = pd.concat(error_dfs, ignore_index=True)
    onnx_df      = run_onnx_interrogation(onnx_paths)

    # ── Print final tradeoff table ────────────────────────────────────────────
    _print_final_tradeoff(tradeoff_df)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    tradeoff_df.to_csv(f"{RESULTS_DIR}/tradeoff.csv",    index=False)
    layer_err_df.to_csv(f"{RESULTS_DIR}/layer_errors.csv", index=False)
    onnx_df.to_csv(f"{RESULTS_DIR}/onnx_graph.csv",      index=False)

    print(f"\n  Results written to {RESULTS_DIR}/")
    print(f"    tradeoff.csv      ({len(tradeoff_df)} rows)")
    print(f"    layer_errors.csv  ({len(layer_err_df)} rows)")
    print(f"    onnx_graph.csv    ({len(onnx_df)} rows)")

    return tradeoff_df, layer_err_df, onnx_df


if __name__ == "__main__":
    main()