"""
quantize/pipeline.py
--------------------
Five-stage post-training quantization (PTQ) pipeline using PyTorch FX graph
mode (prepare_fx / convert_fx), the appropriate quantization API for
PyTorch 2.11.0.

API selection rationale:
  torch.ao.quantization submodules confirmed present in this environment:
    quantize_fx  ✓  — used here
    quantize_pt2e ✗ — not present in this build (requires torchao or ≥ 2.4+)
  FX graph mode is the stable, fully-supported path for this install.

Pipeline stages and their ordering constraint:
  1. fold_batchnorm   — deep-copy + eval; BN folding handled by prepare_fx
  2. insert_observers — prepare_fx(); graph capture + observer insertion
  3. calibrate        — 100 forward passes; observers accumulate activation ranges
  4. convert_to_int8  — convert_fx(); freeze ranges, lower to INT8 ops
  5. export_onnx      — torch.onnx.export(); portable IR for hardware compilers

Design decisions (proposal, Section 6):
  - PTQ over QAT: no training pipeline access, one-day constraint
  - Per-channel symmetric INT8 for weights: distributions vary across channels
  - Per-tensor affine uint8 for activations: uniform within a layer;
    avoids per-channel scale vectors at inference
  - MinMaxObserver as baseline: deterministic, no hyperparameter tuning;
    sensitivity to outliers is a feature of the miscalibration experiment
  - ONNX opset 17: hardware-compiler interchange format
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.fx

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


# ─── Shared QConfig ───────────────────────────────────────────────────────────

def _build_qconfig_mapping() -> QConfigMapping:
    """
    Construct the global QConfigMapping used across all pipeline runs.

    QConfig decision (proposal, Section 6):
      - Per-channel MinMaxObserver for weights: weight distributions vary
        substantially across output channels; per-channel quantization is
        critical for weight accuracy.
      - Per-tensor MinMaxObserver for activations: activation distributions
        are more uniform within a layer; per-tensor avoids the runtime cost
        of per-channel scale vectors at inference.
      - MinMax (not percentile or KL): deterministic, no hyperparameter tuning.
        Sensitivity to outliers is a feature of the miscalibration experiment,
        not a defect. Percentile and KL observers are a natural next step,
        explicitly deferred (proposal, Section 6).
    """
    qconfig = torch.ao.quantization.QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    return QConfigMapping().set_global(qconfig)


# ─── Stage 1: BN-folding ──────────────────────────────────────────────────────

def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    Deep-copy the model and set eval mode.

    BatchNorm folding is handled automatically during prepare_fx() graph
    capture — it is part of the FX tracing pass. This function exists as
    an explicit pipeline stage for two reasons:

      1. It deep-copies the model before any mutation, preserving the original
         FP32 weights in the zoo as the reference signal for layer-wise error
         attribution (error_analysis.py). Mutating the zoo entry in-place
         would destroy that reference.

      2. It documents the stage boundary: the model returned here is a clean
         FP32 copy in eval mode, ready for FX graph capture.

    The BN-folding formula (proposal, Section 4.2):
        W' = (γ / sqrt(σ² + ε)) · W
        b' = γ · (b - μ) / sqrt(σ² + ε) + β
    This happens inside prepare_fx() — we do not implement it manually.
    """
    model = copy.deepcopy(model)
    model.eval()
    return model


# ─── Stage 2: Observer insertion ─────────────────────────────────────────────

def insert_observers(
    model: nn.Module,
    example_input: torch.Tensor,
) -> torch.fx.GraphModule:
    """
    Trace the model into an FX graph and insert MinMax observer nodes.

    prepare_fx() does two things in one pass:
      a) Traces the model into an FX graph IR. BatchNorm folding and
         constant folding are applied during this capture. The result is
         a GraphModule where every operation is explicit and inspectable.
      b) Inserts observer nodes at every quantizable activation site.
         The returned model still computes in FP32; observers are passengers
         that accumulate statistics during calibration.

    Why FX over eager mode:
      FX graph capture means observer insertion, calibration, and quantization
      all operate on the same graph representation. This makes the exported
      ONNX graph more faithful to what was actually quantized — important for
      the ONNX interrogation stage (onnx_interrogate.py).

    Args:
        model:         FP32 model returned by fold_batchnorm().
        example_input: One representative input tensor. Shape matters for
                       graph tracing; values are irrelevant.

    Returns:
        A prepared torch.fx.GraphModule with observer nodes inserted.
    """
    model.eval()
    qconfig_mapping = _build_qconfig_mapping()
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=(example_input,))
    return prepared


# ─── Stage 3: Calibration ─────────────────────────────────────────────────────

def calibrate(
    model: torch.fx.GraphModule,
    calibration_loader,
    n_batches: int = 100,
    device: str = "cpu",
) -> torch.fx.GraphModule:
    """
    Run forward passes so observers accumulate true activation ranges.

    This stage is the intervention point for the miscalibration experiment
    (experiments/miscalibration.py). The entire pipeline is held constant
    across the two experimental conditions; only the calibration DataLoader
    changes — real images versus random Gaussian tensors. That single-variable
    design makes the accuracy delta between the two conditions interpretable
    as the isolated effect of calibration data quality.

    n_batches=100 is a standard PTQ heuristic. The model is not being trained:
    no gradients are computed, no weights are updated. Each observer accumulates
    the min and max activation value seen at its site, producing [α, β] — the
    range that defines the INT8 grid for that layer:

        s = (β - α) / (q_max - q_min)
        z = clip(round(-α / s), q_min, q_max)

    Args:
        model:              Prepared model from insert_observers().
        calibration_loader: DataLoader yielding (images, labels). Labels unused.
        n_batches:          Calibration batches to run. Default: 100.
        device:             Compute device. Default: "cpu".

    Returns:
        The same model with observers populated.
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= n_batches:
                break
            images = images.to(device)
            model(images)
            if (i + 1) % 10 == 0:
                print(f"  Calibration: {i + 1}/{n_batches} batches", flush=True)

    return model


# ─── Stage 4: INT8 conversion ─────────────────────────────────────────────────

def convert_to_int8(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Freeze observed activation ranges and lower the FX graph to INT8 ops.

    convert_fx() does three things in one pass:
      1. Reads the accumulated [α, β] from each observer node.
      2. Computes scale s and zero-point z for each quantizable site.
      3. Replaces observer nodes and FP32 ops with their INT8 equivalents.

    The returned model computes in INT8 (weights and activations) and
    accumulates into INT32 before requantizing — the integer arithmetic
    pipeline of Jacob et al. 2018 (cited in the proposal, Section 4.2).

    This function consumes the prepared model. If the prepared model is
    needed again (e.g. for a second calibration run in the miscalibration
    experiment), deep-copy it before calling this function.

    Returns:
        A quantized torch.fx.GraphModule operating in INT8.
    """
    int8_model = convert_fx(model)
    int8_model.eval()
    return int8_model


# ─── Stage 5: ONNX export ─────────────────────────────────────────────────────

def export_onnx(
    model: torch.fx.GraphModule,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 17,
) -> str:
    """
    Serialize the quantized FX graph to ONNX opset 17.

    Why ONNX opset 17 (proposal, Section 4.2):
      Hardware compilers in the automotive and embedded sector — including
      NXP's toolchain — consume ONNX as their interchange format. Exporting
      to ONNX is not a formality: it is where PyTorch's internal quantized
      representation is lowered to a standardised operator graph. Operators
      that PyTorch can quantize but the ONNX backend cannot represent will
      appear as FP32 nodes in the exported graph. The ONNX interrogation
      stage (onnx_interrogate.py) detects exactly this — counting quantized
      versus FP32 operator nodes is a diagnostic step, not a verification step.

    Why dynamo=False:
      The new dynamo-based ONNX exporter (default in PyTorch 2.x) does not
      support quantized models with packed conv parameters
      (Conv2dPackedParamsBase). The legacy exporter handles these correctly.

    Args:
        model:          Quantized model from convert_to_int8().
        input_shape:    Tensor shape excluding batch dim, e.g. (3, 224, 224).
        output_path:    Destination path for the .onnx file.
        opset_version:  ONNX opset. Default: 17.

    Returns:
        output_path string for downstream consumption by the pipeline runner.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, *input_shape)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            opset_version=opset_version,
            dynamo=False,          # legacy exporter — required for quantized models
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    print(f"  ONNX export: {output_path} (opset {opset_version})")
    return output_path


# ─── Pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(
    entry,
    calibration_loader,
    output_dir: str = "onnx_models",
    n_cal_batches: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Execute all five stages in order and return a results dict.

    This is the entry point called by the benchmark script and the
    miscalibration experiment. It enforces stage ordering, preserves the
    original FP32 model for downstream comparison, and collects all
    artifacts produced by the pipeline.

    Stage ordering is a hard constraint:
      - Stage 1 before Stage 2: deep-copy before any graph mutation
      - Stage 2 before Stage 3: observers must exist before calibration runs
      - Stage 3 before Stage 4: ranges must be populated before conversion
      - Stage 4 before Stage 5: model must be INT8 before ONNX lowering

    Args:
        entry:              ModelEntry from zoo.py.
        calibration_loader: DataLoader for calibration (real or synthetic).
        output_dir:         Directory for exported ONNX files.
        n_cal_batches:      Number of calibration batches. Default: 100.
        device:             Compute device. Default: "cpu".

    Returns:
        {
            "name":       str           — model name, for result tables
            "fp32_model": nn.Module     — original FP32 model, untouched
            "int8_model": GraphModule   — quantized model
            "onnx_path":  str           — path to exported ONNX file
        }
    """
    name = entry.name
    print(f"\n── Pipeline: {name} ──")

    # Preserve the original FP32 model for layer-wise error attribution.
    fp32_model = copy.deepcopy(entry.model)
    fp32_model.eval()

    example_input = torch.zeros(1, *entry.input_shape)

    print("  Stage 1: BN-folding prep (deep-copy + eval)")
    fused = fold_batchnorm(entry.model)

    print("  Stage 2: FX graph capture + observer insertion (prepare_fx)")
    prepared = insert_observers(fused, example_input)

    print("  Stage 3: Calibration")
    calibrated = calibrate(
        prepared,
        calibration_loader,
        n_batches=n_cal_batches,
        device=device,
    )

    print("  Stage 4: INT8 conversion (convert_fx)")
    int8_model = convert_to_int8(calibrated)

    print("  Stage 5: ONNX export (opset 17)")
    onnx_path = export_onnx(
        int8_model,
        input_shape=entry.input_shape,
        output_path=f"{output_dir}/{name}_int8.onnx",
    )

    return {
        "name":       name,
        "fp32_model": fp32_model,
        "int8_model": int8_model,
        "onnx_path":  onnx_path,
    }