"""
quantize/error_analysis.py
--------------------------
Layer-wise quantization error attribution.

Attaches forward hooks to the FP32 and INT8 models simultaneously,
runs the same input through both, and computes per-layer L2 and
max-error metrics. Layers are ranked by L2 error descending.

This is the primary evidence for Hypothesis H1 (proposal, Section 5):
  If depthwise-separable layers in MobileNetV2 rank highest in L2 error
  consistently across both calibration conditions, the result is consistent
  with the hypothesis that reduced weight redundancy in depthwise-separable
  convolutions increases sensitivity to the INT8 grid resolution.

Note on causal language: this experiment surfaces patterns, it does not
establish causation. "Consistent with the hypothesis" is the correct
framing; "proves" is not.
"""

import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List


# ─── Hook-based activation collector ─────────────────────────────────────────

def collect_layer_outputs(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Run one forward pass and collect the output tensor of every named module.

    Uses PyTorch forward hooks — registered callbacks that intercept module
    outputs without modifying the model or its computation. Each hook stores
    a detached CPU copy of the output tensor, keyed by module name.

    Why detached CPU copies:
      - detach(): we are not computing gradients; keeping the computation
        graph would waste memory.
      - .cpu(): hooks fire during the forward pass; copying to CPU immediately
        avoids holding GPU memory after the pass completes.
      - clone(): without clone(), the stored tensor is a view of a buffer
        that may be overwritten by a subsequent operation.

    Why named_modules() and not named_children():
      named_children() returns only top-level submodules. named_modules()
      recurses into every level of nesting — necessary to reach individual
      Conv layers inside MobileNetV2's InvertedResidual blocks and
      EfficientNet-B0's MBConv blocks.

    Args:
        model:        Any nn.Module (FP32 or INT8).
        input_tensor: One input batch. Shape must match model's input_shape.
        device:       Compute device. Default: "cpu".

    Returns:
        Dict mapping module_name → output Tensor (detached, on CPU).
    """
    outputs: Dict[str, torch.Tensor] = {}
    hooks   = []

    def _make_hook(name: str):
        def _hook(module, input, output): # type: ignore[reportUnusedParameter], module and input required by PyTorch hook API
            # Guard: some modules output tuples (e.g. LSTM). Take first element.
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu().clone()
        return _hook

    # Register one hook per named module.
    for name, module in model.named_modules():
        if name == "":
            continue   # skip the root module itself
        hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    model.to(device)

    with torch.no_grad():
        model(input_tensor.to(device))

    # Remove all hooks immediately — leaving them registered would slow down
    # every subsequent forward pass on this model.
    for h in hooks:
        h.remove()

    return outputs


# ─── Error computation ────────────────────────────────────────────────────────

def compute_layer_errors(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Compute per-layer L2 and max quantization error between FP32 and INT8.

    Runs collect_layer_outputs() on both models with the same input,
    then for every layer present in both output dicts computes:

        L2 error(i)  = ‖y_i^fp32 − y_i^int8‖₂
        max error(i) = ‖y_i^fp32 − y_i^int8‖∞

    Layers present in only one model are skipped — FX graph mode may
    introduce or remove wrapper modules, so name alignment is not guaranteed
    to be perfect across FP32 and INT8 representations.

    The result DataFrame is sorted by l2_error descending. This ranking
    is the evidence for H1: if depthwise-separable layers appear at the
    top consistently, the pattern is consistent with the hypothesis.

    Args:
        fp32_model:   Original FP32 model (from run_pipeline "fp32_model").
        int8_model:   Quantized model (from run_pipeline "int8_model").
        input_tensor: One representative input batch.
        device:       Compute device. Default: "cpu".

    Returns:
        pd.DataFrame with columns:
            layer       — module name
            l2_error    — L2 norm of output difference
            max_error   — L∞ norm of output difference
            fp32_shape  — output shape under FP32
    """
    print("  Collecting FP32 layer outputs...")
    fp32_outputs = collect_layer_outputs(fp32_model, input_tensor, device)

    print("  Collecting INT8 layer outputs...")
    int8_outputs = collect_layer_outputs(int8_model, input_tensor, device)

    # Only compare layers present in both models.
    common_layers = set(fp32_outputs.keys()) & set(int8_outputs.keys())
    print(f"  Common layers: {len(common_layers)} "
          f"(FP32: {len(fp32_outputs)}, INT8: {len(int8_outputs)})")

    records: List[dict] = []

    for name in sorted(common_layers):
        y_fp32 = fp32_outputs[name].float()
        y_int8 = int8_outputs[name].float()

        # Shape mismatch can occur if quantization changes a layer's output
        # layout. Skip rather than error — note it for investigation.
        if y_fp32.shape != y_int8.shape:
            print(f"  Shape mismatch at {name}: "
                  f"fp32={y_fp32.shape} int8={y_int8.shape} — skipped")
            continue

        diff      = y_fp32 - y_int8
        l2_error  = float(torch.norm(diff, p=2))
        max_error = float(torch.norm(diff, p=float("inf")))

        records.append({
            "layer":      name,
            "l2_error":   l2_error,
            "max_error":  max_error,
            "fp32_shape": str(tuple(y_fp32.shape)),
        })

    df = pd.DataFrame(records).sort_values("l2_error", ascending=False)
    df = df.reset_index(drop=True)
    return df


# ─── Display ──────────────────────────────────────────────────────────────────

def print_error_table(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print the top-N most sensitive layers ranked by L2 error.

    This table is one of three result tables in the engineering report
    (proposal, Section 4.4). It is the artifact that sustains the H1
    discussion in a technical interview:

      "The depthwise layers in MobileNetV2 ranked highest in L2 error
       consistently across both calibration conditions. That pattern is
       consistent with the hypothesis that reduced weight redundancy
       increases sensitivity to the INT8 grid — though this experiment
       surfaces the pattern, it does not establish causation."

    Args:
        df:    DataFrame from compute_layer_errors(), sorted by l2_error.
        top_n: Number of layers to display. Default: 10.
    """
    print(f"\n── Top-{top_n} layers by L2 quantization error ──")
    print(f"  {'Rank':<5} {'L2 error':>10} {'Max error':>10}  Layer")
    print(f"  {'─'*5} {'─'*10} {'─'*10}  {'─'*50}")

    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        print(
         f"  {rank:<5} "
         f"{row['l2_error']:>10.4f} "
         f"{row['max_error']:>10.4f}  "
         f"{row['layer']}"
        )

    print(f"\n  Total layers compared: {len(df)}")


# ─── Top-level entry point ────────────────────────────────────────────────────

def run_error_attribution(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    entry,
    device: str = "cpu",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Run the full layer-wise error attribution for one model.

    Entry point called by the benchmark script and the miscalibration
    experiment. Constructs a representative input from the ModelEntry's
    input_shape, runs both models, computes the error table, prints the
    top-N ranking, and returns the full DataFrame for CSV export.

    Args:
        fp32_model: Original FP32 model.
        int8_model: Quantized INT8 model.
        entry:      ModelEntry from zoo.py (used for input_shape and name).
        device:     Compute device. Default: "cpu".
        top_n:      Layers to display in the printed table. Default: 10.

    Returns:
        Full error DataFrame sorted by l2_error descending.
    """
    print(f"\n── Error attribution: {entry.name} ──")

    # Use a fixed-seed random input so results are reproducible across runs.
    # The input distribution does not affect the attribution: we are measuring
    # the difference between two deterministic functions (FP32 and INT8)
    # on the same input, not characterising the input distribution.
    torch.manual_seed(42)
    input_tensor = torch.randn(1, *entry.input_shape)

    df = compute_layer_errors(fp32_model, int8_model, input_tensor, device)
    print_error_table(df, top_n=top_n)

    return df