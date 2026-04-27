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

QConfig decision:
  Uses get_default_qconfig_mapping("fbgemm") — PyTorch's recommended
  production PTQ configuration. This uses HistogramObserver for activations
  (percentile-based, outlier-robust) and PerChannelMinMaxObserver for weights.

  The original custom MinMaxObserver QConfig was replaced after discovering
  that MinMaxObserver causes quantization collapse on MobileNetV2 and
  EfficientNet-B0: extreme activation outliers stretch the INT8 grid so far
  that the model predicts a single class for all inputs. HistogramObserver
  clips outliers at the 99.99th percentile, producing stable grids.

  MinMaxObserver is retained in experiments/miscalibration.py where its
  sensitivity to outliers is the controlled variable being studied (H3).

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
from torch.ao.quantization import get_default_qconfig_mapping


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
) -> torch.fx.GraphModule:
    """
    Trace the model into an FX graph and insert observer nodes.

    Uses get_default_qconfig_mapping("fbgemm") — PyTorch's recommended
    production PTQ configuration. "fbgemm" targets x86 CPUs and uses
    HistogramObserver for activations, which clips outliers at the 99.99th
    percentile rather than tracking absolute min/max.

    Why not the custom MinMaxObserver QConfig:
      MinMaxObserver causes quantization collapse on MobileNetV2 and
      EfficientNet-B0 — a single activation outlier stretches the INT8 grid
      to cover values that never appear at inference, wasting most of the
      256 steps. HistogramObserver is robust to this.

    Args:
        model:         FP32 model returned by fold_batchnorm().
        example_input: One representative input tensor.

    Returns:
        A prepared torch.fx.GraphModule with observer nodes inserted.
    """
    model.eval()
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
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

    The model runs in FP32 during calibration — no gradients, no weight
    updates. Observers accumulate statistics that define the INT8 grid
    per layer. With HistogramObserver, the grid is defined by the
    99.99th percentile of observed activations, not the absolute max.

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

    convert_fx() reads observer statistics, computes scale s and zero-point z
    per layer, and replaces FP32 ops with INT8 equivalents. The model
    accumulates into INT32 before requantizing — the integer arithmetic
    pipeline of Jacob et al. 2018.

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
      The dynamo-based ONNX exporter (default in PyTorch 2.x) does not
      support quantized models with Conv2dPackedParamsBase. The legacy
      exporter handles these correctly.

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

    Returns:
        {
            "name":       str        — model name
            "fp32_model": nn.Module  — original FP32 model, untouched
            "int8_model": GraphModule — quantized model
            "onnx_path":  str        — path to exported ONNX file
        }
    """
    name = entry.name
    print(f"\n── Pipeline: {name} ──")

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