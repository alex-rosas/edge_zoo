"""
quantize/pipeline.py
--------------------
Five-stage post-training quantization (PTQ) pipeline using PyTorch FX graph
mode (prepare_fx / convert_fx).

Pipeline stages and their ordering constraint:
  1. fold_batchnorm   — deep-copy + eval; BN folding handled by prepare_fx
  2. insert_observers — prepare_fx(); graph capture + observer insertion
  3. calibrate        — 100 forward passes; observers accumulate activation ranges
  4. convert_to_int8  — convert_fx(); freeze ranges, lower to INT8 ops
  5. export_onnx      — torch.onnx.export(); portable IR for hardware compilers

QConfig options (qconfig parameter in run_pipeline / insert_observers):
  "fbgemm"  — get_default_qconfig_mapping("fbgemm"). HistogramObserver for
               activations (percentile-based, outlier-robust). PyTorch's
               recommended production PTQ configuration for x86 CPUs.
               Default for the main pipeline.

  "minmax"  — custom QConfig with MinMaxObserver for activations and
               PerChannelMinMaxObserver for weights. Deterministic, no
               hyperparameters. Sensitive to outliers — causes quantization
               collapse on MobileNetV2 and EfficientNet-B0 under CIFAR-10
               calibration. Retained for the observer comparison experiment
               (experiments/observer_comparison.py) where observer sensitivity
               is the variable under study.

Design decisions:
  - PTQ over QAT: no training pipeline access, one-day constraint
  - ONNX opset 17: hardware-compiler interchange format
  - dynamo=False: legacy exporter required for quantized Conv2dPackedParamsBase
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.fx

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
from torch.ao.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


# ─── QConfig factory ──────────────────────────────────────────────────────────

def _get_qconfig_mapping(qconfig: str) -> QConfigMapping:
    """
    Return a QConfigMapping by name.

    Args:
        qconfig: "fbgemm" — HistogramObserver (production default)
                 "minmax" — MinMaxObserver (observer comparison experiment)

    Raises:
        ValueError: if qconfig is not one of the two supported values.
    """
    if qconfig == "fbgemm":
        return get_default_qconfig_mapping("fbgemm")

    if qconfig == "minmax":
        qc = torch.ao.quantization.QConfig(
            activation=MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            ),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            ),
        )
        return QConfigMapping().set_global(qc)

    raise ValueError(
        f"Unknown qconfig '{qconfig}'. Choose 'fbgemm' or 'minmax'."
    )


# ─── Stage 1: BN-folding ──────────────────────────────────────────────────────

def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    Deep-copy the model and set eval mode.

    BatchNorm folding is handled automatically during prepare_fx() graph
    capture. This function exists as an explicit stage to:
      1. Deep-copy the model before any mutation, preserving the original
         FP32 weights for layer-wise error attribution.
      2. Document the stage boundary clearly.
    """
    model = copy.deepcopy(model)
    model.eval()
    return model


# ─── Stage 2: Observer insertion ─────────────────────────────────────────────

def insert_observers(
    model: nn.Module,
    example_input: torch.Tensor,
    qconfig: str = "fbgemm",
) -> torch.fx.GraphModule:
    """
    Trace the model into an FX graph and insert observer nodes.

    Args:
        model:         FP32 model returned by fold_batchnorm().
        example_input: One representative input tensor.
        qconfig:       Observer configuration. "fbgemm" (default) or "minmax".

    Returns:
        A prepared torch.fx.GraphModule with observer nodes inserted.
    """
    model.eval()
    qconfig_mapping = _get_qconfig_mapping(qconfig)
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
    Run forward passes so observers accumulate activation statistics.

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

    Why dynamo=False:
      The dynamo-based ONNX exporter does not support quantized models with
      Conv2dPackedParamsBase. The legacy exporter handles these correctly.

    Args:
        model:         Quantized model from convert_to_int8().
        input_shape:   Tensor shape excluding batch dim, e.g. (3, 224, 224).
        output_path:   Destination path for the .onnx file.
        opset_version: ONNX opset. Default: 17.

    Returns:
        output_path string.
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
            dynamo=False,
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
    qconfig: str = "fbgemm",
) -> dict:
    """
    Execute all five stages in order and return a results dict.

    Stage ordering is a hard constraint:
      - Stage 1 before Stage 2: deep-copy before any graph mutation
      - Stage 2 before Stage 3: observers must exist before calibration runs
      - Stage 3 before Stage 4: ranges must be populated before conversion
      - Stage 4 before Stage 5: model must be INT8 before ONNX lowering

    Args:
        entry:              ModelEntry from zoo.py.
        calibration_loader: DataLoader for calibration.
        output_dir:         Directory for exported ONNX files.
        n_cal_batches:      Number of calibration batches. Default: 100.
        device:             Compute device. Default: "cpu".
        qconfig:            Observer configuration. "fbgemm" (default) or
                            "minmax". Pass "minmax" for the observer comparison
                            experiment.

    Returns:
        {
            "name":       str         — model name
            "fp32_model": nn.Module   — original FP32 model, untouched
            "int8_model": GraphModule — quantized model
            "onnx_path":  str         — path to exported ONNX file
            "qconfig":    str         — observer used, for result table labelling
        }
    """
    name = entry.name
    print(f"\n── Pipeline: {name} (qconfig={qconfig}) ──")

    fp32_model = copy.deepcopy(entry.model)
    fp32_model.eval()

    example_input = torch.zeros(1, *entry.input_shape)

    print("  Stage 1: BN-folding prep (deep-copy + eval)")
    fused = fold_batchnorm(entry.model)

    print(f"  Stage 2: FX graph capture + observer insertion ({qconfig})")
    prepared = insert_observers(fused, example_input, qconfig=qconfig)

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
        output_path=f"{output_dir}/{name}_{qconfig}_int8.onnx",
    )

    return {
        "name":       name,
        "fp32_model": fp32_model,
        "int8_model": int8_model,
        "onnx_path":  onnx_path,
        "qconfig":    qconfig,
    }