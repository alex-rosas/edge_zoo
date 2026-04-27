"""
experiments/observer_comparison.py
------------------------------------
Controlled experiment: MinMaxObserver vs HistogramObserver (fbgemm default).

Everything held constant across the two conditions:
  - Architecture (same three models)
  - Calibration data (CIFAR-10, 100 batches)
  - Evaluation data (same preloaded ImageNet batches)
  - Bit-width (INT8)
  - ONNX opset (17)

Single free variable: observer type.

Why this experiment exists:
  During Phase 2, MinMaxObserver caused quantization collapse on MobileNetV2
  and EfficientNet-B0 under CIFAR-10 calibration — the model predicted a
  single class for all inputs. The pipeline was switched to
  get_default_qconfig_mapping("fbgemm") (HistogramObserver) which resolved
  the collapse. This experiment documents the difference quantitatively and
  identifies which architectures are sensitive to observer choice.

Expected finding:
  ResNet-18 (residual, standard Conv) is robust to both observers.
  MobileNetV2 (depthwise-separable) and EfficientNet-B0 (compound-scaled,
  Sigmoid-gated SE blocks) show disproportionate sensitivity to MinMaxObserver.
  This is consistent with the hypothesis that reduced weight redundancy and
  non-standard activation patterns increase observer sensitivity.

Output:
  results/observer_comparison.csv  — accuracy delta per model per observer
"""

import sys
from pathlib import Path

import torch
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.zoo import get_zoo, print_zoo_summary
from quantize.pipeline import run_pipeline
from quantize.observers import (
    real_calibration_loader,
    imagenet_eval_loader,
    preload_eval_batches,
    evaluate_top1,
)


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR       = "./data"
OUTPUT_DIR     = "onnx_models/observer_comparison"
RESULTS_DIR    = "results"
N_CAL_BATCHES  = 100
N_EVAL_BATCHES = 63      # 63 × 32 = 2016 ImageNet val images


# ─── Per-model, per-observer run ──────────────────────────────────────────────

def run_one(entry, cal_loader, eval_batches, qconfig: str) -> dict:
    """
    Run the full PTQ pipeline for one model under one observer and
    return scalar results.
    """
    result = run_pipeline(
        entry,
        calibration_loader=cal_loader,
        output_dir=OUTPUT_DIR,
        n_cal_batches=N_CAL_BATCHES,
        device="cpu",
        qconfig=qconfig,
    )

    fp32_model = result["fp32_model"]
    int8_model = result["int8_model"]

    fp32_top1 = evaluate_top1(fp32_model, eval_batches, device="cpu",
                               max_batches=N_EVAL_BATCHES)
    int8_top1 = evaluate_top1(int8_model, eval_batches, device="cpu",
                               max_batches=N_EVAL_BATCHES)

    return {
        "model":          entry.name,
        "arch_family":    entry.arch_family,
        "qconfig":        qconfig,
        "fp32_top1":      fp32_top1,
        "int8_top1":      int8_top1,
        "accuracy_delta": fp32_top1 - int8_top1,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a side-by-side comparison of fbgemm vs minmax per model.
    """
    print(f"\n{'═'*80}")
    print(" OBSERVER COMPARISON TABLE")
    print(f"{'═'*80}")
    print(f"  {'Model':<20} {'FP32':>8} {'INT8 fbgemm':>13} {'Δ fbgemm':>10} "
          f"{'INT8 minmax':>13} {'Δ minmax':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*13} {'─'*10} {'─'*13} {'─'*10}")

    models = df["model"].unique()
    for model in models:
        fbgemm = df[(df["model"] == model) & (df["qconfig"] == "fbgemm")].iloc[0]
        minmax = df[(df["model"] == model) & (df["qconfig"] == "minmax")].iloc[0]

        print(
            f"  {model:<20} "
            f"{fbgemm['fp32_top1']*100:>7.2f}% "
            f"{fbgemm['int8_top1']*100:>12.2f}% "
            f"{fbgemm['accuracy_delta']*100:>+9.2f}% "
            f"{minmax['int8_top1']*100:>12.2f}% "
            f"{minmax['accuracy_delta']*100:>+9.2f}%"
        )

    print(f"\n  Δ = FP32 top-1 − INT8 top-1 (positive = accuracy lost to quantization)")
    print(f"  fbgemm = HistogramObserver (outlier-robust, percentile clipping)")
    print(f"  minmax = MinMaxObserver (deterministic, sensitive to outliers)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n══ EdgeZoo: Observer Comparison Experiment ══")
    print("  Variable: observer type (fbgemm vs minmax)")
    print("  Held constant: architecture, calibration data, evaluation data,")
    print("                 bit-width, ONNX opset\n")
    print_zoo_summary()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    zoo = get_zoo()

    # Preload eval batches once — both observers evaluate on identical images.
    print("\n  Building ImageNet evaluation loader...")
    eval_loader  = imagenet_eval_loader(batch_size=32, max_samples=2016)
    eval_batches = preload_eval_batches(eval_loader, max_batches=N_EVAL_BATCHES)

    records = []

    for entry in zoo.values():
        print(f"\n{'─'*60}")
        print(f" Model: {entry.name}")
        print(f"{'─'*60}")

        # Fresh calibration loader for each run
        cal_loader = real_calibration_loader(
            data_dir=DATA_DIR, batch_size=32,
            image_size=entry.input_shape[1],
        )

        print("\n  [1/2] Running fbgemm (HistogramObserver)...")
        records.append(run_one(entry, cal_loader, eval_batches, "fbgemm"))

        # Re-create calibration loader — it was consumed in the previous run
        cal_loader = real_calibration_loader(
            data_dir=DATA_DIR, batch_size=32,
            image_size=entry.input_shape[1],
        )

        print("\n  [2/2] Running minmax (MinMaxObserver)...")
        records.append(run_one(entry, cal_loader, eval_batches, "minmax"))

    df = pd.DataFrame(records)
    print_comparison_table(df)

    out_path = f"{RESULTS_DIR}/observer_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Results written to {out_path}\n")

    return df


if __name__ == "__main__":
    main()