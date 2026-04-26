"""
benchmark/bench.py
------------------
Main benchmark script: runs all three models through the PTQ pipeline
and produces the three result tables for the engineering report.

Output files:
  results/tradeoff.csv     — accuracy delta, size, latency per model
  results/layer_errors.csv — layer-wise L2 and max error per model
  results/onnx_graph.csv   — quantized vs FP32 operator counts per model

Device:
  Uses CUDA if available, falls back to CPU.
  Latency measured with CUDA events on GPU for accuracy.

Evaluation dataset:
  evanarlian/imagenet_1k_resized_256 via HF streaming.
  No token or licence required. Produces top-1 numbers directly
  comparable to published benchmarks.

Calibration dataset:
  CIFAR-10 (unchanged) — used only for observer calibration, not evaluation.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from models.zoo import get_zoo, print_zoo_summary
from quantize.pipeline import run_pipeline
from quantize.observers import (
    real_calibration_loader,
    imagenet_eval_loader,
    evaluate_top1,
)
from quantize.error_analysis import run_error_attribution
from quantize.onnx_interrogate import run_onnx_interrogation


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR       = "./data"
OUTPUT_DIR     = "onnx_models"
RESULTS_DIR    = "results"
N_LATENCY_RUNS = 50
N_CAL_BATCHES  = 100
N_EVAL_BATCHES = 63       # 63 × 32 = 2016 ImageNet val images
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Latency measurement ──────────────────────────────────────────────────────

def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    n_runs: int = N_LATENCY_RUNS,
    device: str = DEVICE,
) -> float:
    """
    Mean inference latency in milliseconds over n_runs repetitions.
    First run excluded as warm-up.
    Uses CUDA events on GPU for accurate timing; perf_counter on CPU.
    """
    model.eval()
    model.to(device)
    dummy = torch.zeros(1, *input_shape).to(device)

    with torch.no_grad():
        model(dummy)   # warm-up

        if device == "cuda":
            times = []
            for _ in range(n_runs):
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                model(dummy)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        else:
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


# ─── Model size measurement ───────────────────────────────────────────────────

def measure_onnx_size_mb(onnx_path: str) -> float:
    return Path(onnx_path).stat().st_size / (1024 ** 2)


# ─── Per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(entry, eval_loader) -> dict:
    name = entry.name
    print(f"\n{'═'*60}")
    print(f" Benchmarking: {name}")
    print(f"{'═'*60}")

    # Calibration — CIFAR-10, CPU
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
        device="cpu",
    )

    fp32_model = pipeline_result["fp32_model"]
    int8_model = pipeline_result["int8_model"]
    onnx_path  = pipeline_result["onnx_path"]

    # Latency on DEVICE
    print(f"\n  Measuring FP32 latency ({N_LATENCY_RUNS} runs on {DEVICE})...")
    fp32_latency = measure_latency(fp32_model, entry.input_shape, device=DEVICE)

    print(f"  Measuring INT8 latency ({N_LATENCY_RUNS} runs on {DEVICE})...")
    int8_latency = measure_latency(int8_model, entry.input_shape, device=DEVICE)

    latency_ratio = fp32_latency / int8_latency if int8_latency > 0 else float("nan")

    fp32_size_mb = entry.size_fp32_mb
    int8_size_mb = measure_onnx_size_mb(onnx_path)
    size_ratio   = fp32_size_mb / int8_size_mb if int8_size_mb > 0 else float("nan")

    # Accuracy on DEVICE with ImageNet
    print(f"  Evaluating FP32 accuracy (ImageNet val, {DEVICE})...")
    fp32_top1 = evaluate_top1(
        fp32_model, eval_loader, device=DEVICE, max_batches=N_EVAL_BATCHES
    )

    print(f"  Evaluating INT8 accuracy (ImageNet val, {DEVICE})...")
    int8_top1 = evaluate_top1(
        int8_model, eval_loader, device=DEVICE, max_batches=N_EVAL_BATCHES
    )

    accuracy_delta = fp32_top1 - int8_top1

    # Error attribution on CPU
    error_df = run_error_attribution(
        fp32_model, int8_model, entry, device="cpu"
    )
    error_df["model"] = name

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
    print(f"  Device: {DEVICE}")
    print_zoo_summary()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    zoo = get_zoo()

    print("\n  Building ImageNet evaluation loader...")
    eval_loader = imagenet_eval_loader(
        batch_size=32,
        max_samples=2016,
    )

    tradeoff_rows = []
    error_dfs     = []
    onnx_paths    = {}

    for entry in zoo.values():
        result = benchmark_model(entry, eval_loader)
        tradeoff_rows.append(result["scalar"])
        error_dfs.append(result["error_df"])
        onnx_paths[entry.name] = result["scalar"]["onnx_path"]

    tradeoff_df  = pd.DataFrame(tradeoff_rows)
    layer_err_df = pd.concat(error_dfs, ignore_index=True)
    onnx_df      = run_onnx_interrogation(onnx_paths)

    _print_final_tradeoff(tradeoff_df)

    tradeoff_df.to_csv(f"{RESULTS_DIR}/tradeoff.csv",     index=False)
    layer_err_df.to_csv(f"{RESULTS_DIR}/layer_errors.csv", index=False)
    onnx_df.to_csv(f"{RESULTS_DIR}/onnx_graph.csv",       index=False)

    print(f"\n  Results written to {RESULTS_DIR}/")
    print(f"    tradeoff.csv      ({len(tradeoff_df)} rows)")
    print(f"    layer_errors.csv  ({len(layer_err_df)} rows)")
    print(f"    onnx_graph.csv    ({len(onnx_df)} rows)")

    return tradeoff_df, layer_err_df, onnx_df


if __name__ == "__main__":
    main()