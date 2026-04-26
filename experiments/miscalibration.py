"""
experiments/miscalibration.py
------------------------------
Controlled miscalibration experiment.

Runs the full PTQ pipeline twice per model — once with real CIFAR-10
calibration images, once with random Gaussian tensors — holding every
other variable constant. The accuracy delta between the two conditions
isolates calibration data quality as the causal variable.

Hypothesis H3 (proposal, Section 5):
  Real-image calibration will outperform Gaussian calibration by at
  least 0.5% top-1 accuracy. The degradation under Gaussian calibration
  will concentrate in early layers, where the Gaussian range estimate
  deviates most from the true activation distribution of natural images.

Experimental controls (proposal, Section 6):
  Architecture, observer type, bit-width, n_batches, and all pipeline
  stages are held constant. Only the calibration DataLoader changes.
  This is the minimal valid experiment for isolating calibration quality
  as a causal factor.

Scientific restraint note:
  Results are reported as "consistent with" or "inconsistent with" H3,
  not as proof. The experiment is designed to surface the pattern;
  establishing the mechanism requires additional controls.
"""

import copy
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List

from quantize.pipeline import (
    fold_batchnorm,
    insert_observers,
    calibrate,
    convert_to_int8,
)
from quantize.observers import (
    real_calibration_loader,
    gaussian_calibration_loader,
    get_observer_ranges,
    evaluate_top1
)
from quantize.error_analysis import run_error_attribution

# ─── Single condition run ─────────────────────────────────────────────────────

def _run_condition(
    entry,
    calibration_loader,
    eval_loader,
    condition_name: str,
    device: str = "cpu",
) -> Dict:
    """
    Run the full pipeline under one calibration condition.

    Executes Stages 1–4 (fold, observe, calibrate, convert), captures
    observer ranges immediately after calibration and before conversion,
    then evaluates accuracy on the shared eval loader.

    Stage 5 (ONNX export) is skipped here — the miscalibration experiment
    compares accuracy and observer ranges, not exported graphs. ONNX export
    happens in the main benchmark script.

    Returns a dict of results for this condition, consumed by
    run_miscalibration_experiment() for side-by-side comparison.
    """
    print(f"\n  ── Condition: {condition_name} ──")
    fp32_model = copy.deepcopy(entry.model)
    fp32_model.eval()

    example_input = torch.zeros(1, *entry.input_shape)

    print("    Stage 1: BN-folding prep")
    fused = fold_batchnorm(entry.model)

    print("    Stage 2: Observer insertion")
    prepared = insert_observers(fused, example_input)

    print("    Stage 3: Calibration")
    calibrated = calibrate(
        prepared,
        calibration_loader,
        n_batches=100,
        device=device,
    )

    # Capture observer ranges before conversion consumes them.
    # After convert_to_int8(), observer nodes are replaced with static
    # scale/zero-point constants — the ranges are no longer accessible.
    print("    Capturing observer ranges...")
    ranges = get_observer_ranges(calibrated)

    print("    Stage 4: INT8 conversion")
    int8_model = convert_to_int8(calibrated)

    print("    Evaluating accuracy...")
    top1 = evaluate_top1(int8_model, eval_loader, device=device)

    print("    Running error attribution...")
    error_df = run_error_attribution(
        fp32_model, int8_model, entry, device=device
    )

    return {
        "condition":   condition_name,
        "int8_model":  int8_model,
        "top1":        top1,
        "ranges":      ranges,
        "error_df":    error_df,
    }


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_miscalibration_experiment(
    entry,
    data_dir: str = "./data",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the controlled miscalibration experiment for one model.

    Builds a shared eval loader (used by both conditions for comparability),
    runs the pipeline under both calibration conditions, prints a side-by-side
    comparison, and returns a summary DataFrame for the results table.

    The shared eval loader is the same CIFAR-10 test split used for
    calibration in the real condition, with a fixed shuffle seed so
    both conditions are evaluated on the same images.

    Args:
        entry:    ModelEntry from zoo.py.
        data_dir: Directory for CIFAR-10 download.
        device:   Compute device. Default: "cpu".

    Returns:
        pd.DataFrame with one row per condition:
            model, condition, top1_accuracy, accuracy_delta,
            n_observers, mean_range_width
    """
    print(f"\n══ Miscalibration experiment: {entry.name} ══")

    # Shared evaluation loader — fixed seed for reproducibility.
    torch.manual_seed(42)
    eval_loader = real_calibration_loader(
        data_dir=data_dir,
        batch_size=32,
        image_size=entry.input_shape[1],
    )

    # Calibration loaders — the single variable that differs.
    real_loader = real_calibration_loader(
        data_dir=data_dir,
        batch_size=32,
        image_size=entry.input_shape[1],
    )
    gauss_loader = gaussian_calibration_loader(
        batch_size=32,
        n_batches=100,
        image_size=entry.input_shape[1],
    )

    # Run both conditions.
    real_result  = _run_condition(
        entry, real_loader,  eval_loader, "real_images", device
    )
    gauss_result = _run_condition(
        entry, gauss_loader, eval_loader, "gaussian",    device
    )

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _print_comparison(entry.name, real_result, gauss_result)

    # ── Observer range comparison ─────────────────────────────────────────────
    _print_range_comparison(real_result["ranges"], gauss_result["ranges"])

    # ── Summary DataFrame ─────────────────────────────────────────────────────
    rows = []
    for result in [real_result, gauss_result]:
        widths = [
            r["max"] - r["min"]
            for r in result["ranges"].values()
        ]
        mean_width = sum(widths) / len(widths) if widths else 0.0

        rows.append({
            "model":           entry.name,
            "condition":       result["condition"],
            "top1_accuracy":   result["top1"],
            "n_observers":     len(result["ranges"]),
            "mean_range_width":mean_width,
        })

    df = pd.DataFrame(rows)

    # Compute accuracy delta: real minus gaussian.
    if len(df) == 2:
        delta = (
            df.loc[df["condition"] == "real_images", "top1_accuracy"].values[0]
            - df.loc[df["condition"] == "gaussian",  "top1_accuracy"].values[0]
        )
        df["accuracy_delta"] = [0.0, -delta]
        print(f"\n  Accuracy delta (real − gaussian): {delta*100:+.2f}%")
        _interpret_h3(delta)

    return df


# ─── Display helpers ──────────────────────────────────────────────────────────

def _print_comparison(
    model_name: str,
    real: Dict,
    gauss: Dict,
) -> None:
    """Print a side-by-side accuracy comparison table."""
    print(f"\n── Results: {model_name} ──")
    print(f"  {'Condition':<20} {'Top-1':>8}")
    print(f"  {'─'*20} {'─'*8}")
    print(f"  {'real_images':<20} {real['top1']*100:>7.2f}%")
    print(f"  {'gaussian':<20} {gauss['top1']*100:>7.2f}%")
    delta = real["top1"] - gauss["top1"]
    print(f"  {'delta':<20} {delta*100:>+7.2f}%")


def _print_range_comparison(
    real_ranges: Dict,
    gauss_ranges: Dict,
    top_n: int = 10,
) -> None:
    """
    Print the layers where Gaussian calibration produced the widest
    range relative to real-image calibration.

    A large ratio (gaussian_width / real_width) means the Gaussian
    input drove activations far outside their true range — the INT8
    grid gets stretched to cover values that never appear at inference.
    This is the mechanistic explanation for any accuracy delta observed.
    """
    common = set(real_ranges.keys()) & set(gauss_ranges.keys())
    if not common:
        print("  No common observer layers to compare.")
        return

    ratios = []
    for name in common:
        r_width = real_ranges[name]["max"]  - real_ranges[name]["min"]
        g_width = gauss_ranges[name]["max"] - gauss_ranges[name]["min"]
        if r_width > 1e-6:
            ratios.append({
                "layer":        name,
                "real_width":   r_width,
                "gauss_width":  g_width,
                "ratio":        g_width / r_width,
            })

    ratios.sort(key=lambda x: x["ratio"], reverse=True)

    print(f"\n  Top-{top_n} layers by Gaussian/real range width ratio:")
    print(f"  {'Layer':<45} {'Real':>8} {'Gauss':>8} {'Ratio':>7}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*7}")

    for r in ratios[:top_n]:
        name = r["layer"][-45:] if len(r["layer"]) > 45 else r["layer"]
        print(
            f"  {name:<45} "
            f"{r['real_width']:>8.3f} "
            f"{r['gauss_width']:>8.3f} "
            f"{r['ratio']:>7.2f}×"
        )


def _interpret_h3(delta: float) -> None:
    """
    Print a scientifically restrained interpretation of the accuracy delta.

    Uses "consistent with" / "inconsistent with" language throughout.
    The experiment surfaces the pattern; it does not establish causation.
    """
    threshold = 0.005   # H3 predicts delta >= 0.5%

    print("\n  H3 interpretation:")
    if delta >= threshold:
        print(
            f"  The accuracy delta ({delta*100:.2f}%) meets the H3 threshold "
            f"of ≥ 0.5%. This result is consistent with H3: real-image "
            f"calibration outperforms Gaussian calibration by a detectable "
            f"margin. The range width ratios above identify which layers "
            f"received the most distorted INT8 grids under Gaussian inputs."
        )
    elif delta > 0:
        print(
            f"  The accuracy delta ({delta*100:.2f}%) is positive but below "
            f"the H3 threshold of 0.5%. Real-image calibration outperforms "
            f"Gaussian, but the margin is smaller than predicted. This result "
            f"is partially consistent with H3 — direction confirmed, magnitude "
            f"not. Possible explanations: the model is robust to range "
            f"distortion at this bit-width, or 100 calibration batches are "
            f"sufficient for MinMaxObserver even with poor data."
        )
    else:
        print(
            f"  The accuracy delta ({delta*100:.2f}%) is zero or negative. "
            f"This result is inconsistent with H3. Possible explanations: "
            f"the evaluation set is too small to detect the difference, or "
            f"the MinMaxObserver is insensitive to calibration data quality "
            f"for this architecture at INT8 resolution."
        )