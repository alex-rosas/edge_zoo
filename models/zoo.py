# models/zoo.py
"""
Model registry for EdgeZoo.

Each entry in the registry is a dataclass that bundles a pre-trained model
with the metadata needed to make an informed selection *before* quantization.
The registry exposes a uniform interface: retrieve by name, query metadata,
apply any benchmark function. Adding a new architecture requires touching
only this file.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict
import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """
    A registered model and its pre-quantization metadata.

    Attributes
    ----------
    name : str
        Registry key. Used to retrieve the entry and label result tables.
    model : nn.Module
        Pre-trained model in eval mode, weights on CPU.
    top1_accuracy : float
        Reported ImageNet top-1 accuracy (%) for the pretrained weights.
        This is the *baseline* against which post-quantization accuracy
        delta is measured.
    n_params_M : float
        Parameter count in millions.  Determines the theoretical INT8 size
        floor: n_params_M * 1 MB (INT8) vs n_params_M * 4 MB (FP32).
    flops_G : float
        Approximate GFLOPs for a single 224×224 inference pass.
        Arithmetic intensity predicts how much a dedicated integer NPU
        (no FP unit, e.g. NXP Ara-2) will benefit from INT8.
    arch_family : str
        Architectural family label.  Records the structural property that
        motivates Hypothesis H1: residual, depthwise-separable, or
        compound-scaled.  Without this label, layer-wise attribution results
        are numbers without an explanatory frame.
    input_shape : tuple
        Expected input tensor shape (C, H, W).  Required by the calibration
        stage, which must construct dummy or real tensors of the right shape.
    """
    name: str
    model: nn.Module
    top1_accuracy: float   # % on ImageNet, pretrained weights
    n_params_M: float      # millions of parameters
    flops_G: float         # GFLOPs at 224x224
    arch_family: str       # "residual" | "depthwise-separable" | "compound-scaled"
    input_shape: tuple = field(default=(3, 224, 224))

    def size_fp32_MB(self) -> float:
        """FP32 model size in MB (4 bytes per parameter)."""
        return self.n_params_M * 4.0

    def size_int8_MB(self) -> float:
        """Theoretical INT8 size in MB (1 byte per parameter)."""
        return self.n_params_M * 1.0

    def summary(self) -> str:
        return (
            f"{self.name:<20} | top-1: {self.top1_accuracy:.1f}%"
            f" | params: {self.n_params_M:.1f}M"
            f" | FP32: {self.size_fp32_MB():.1f}MB"
            f" | INT8 (theoretical): {self.size_int8_MB():.1f}MB"
            f" | FLOPs: {self.flops_G:.1f}G"
            f" | family: {self.arch_family}"
        )


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def _build_registry() -> Dict[str, ModelEntry]:
    """
    Instantiate and register all three zoo models.

    Pre-trained weights are downloaded from torchvision on first call and
    cached locally.  All models are placed in eval mode on CPU — quantization
    runs on CPU because torch.ao.quantization does not support CUDA backends
    for static PTQ.
    """
    registry: Dict[str, ModelEntry] = {}

    # --- ResNet-18 ---
    # Residual connections make this a natural baseline: deep, redundant
    # weights, well-studied quantization behaviour.
    resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    resnet.eval()
    registry["resnet18"] = ModelEntry(
        name="resnet18",
        model=resnet,
        top1_accuracy=69.8,
        n_params_M=11.7,
        flops_G=1.8,
        arch_family="residual",
    )

    # --- MobileNetV2 ---
    # Depthwise-separable convolutions reduce parameter redundancy.
    # This is the key structural property for Hypothesis H1: each weight
    # is more "load-bearing", which may increase INT8 grid sensitivity.
    mobilenet = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet.eval()
    registry["mobilenetv2"] = ModelEntry(
        name="mobilenetv2",
        model=mobilenet,
        top1_accuracy=71.9,
        n_params_M=3.4,
        flops_G=0.3,
        arch_family="depthwise-separable",
    )

    # --- EfficientNet-B0 ---
    # Compound scaling (depth × width × resolution) produces heterogeneous
    # activation distributions across layers — the proposal predicts this
    # makes its quantization behaviour less predictable than either baseline.
    efficientnet = tvm.efficientnet_b0(
        weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    efficientnet.eval()
    registry["efficientnet_b0"] = ModelEntry(
        name="efficientnet_b0",
        model=efficientnet,
        top1_accuracy=77.7,
        n_params_M=5.3,
        flops_G=0.4,
        arch_family="compound-scaled",
    )

    return registry


# Module-level singleton — built once, shared across all callers.
_REGISTRY: Dict[str, ModelEntry] = _build_registry()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_model(name: str) -> ModelEntry:
    """Retrieve a registered model entry by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_REGISTRY.keys())


def print_zoo_summary() -> None:
    """Print the pre-quantization metadata table for all registered models."""
    print("\n── EdgeZoo: Pre-quantization model summary ──")
    print(f"{'Model':<20} | {'Top-1':>6} | {'Params':>7} | "
          f"{'FP32 MB':>8} | {'INT8 MB':>8} | {'GFLOPs':>7} | Family")
    print("─" * 85)
    for entry in _REGISTRY.values():
        print(
            f"{entry.name:<20} | {entry.top1_accuracy:>5.1f}% "
            f"| {entry.n_params_M:>6.1f}M "
            f"| {entry.size_fp32_MB():>7.1f}MB "
            f"| {entry.size_int8_MB():>7.1f}MB "
            f"| {entry.flops_G:>6.1f}G "
            f"| {entry.arch_family}"
        )
    print()


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to verify the registry builds cleanly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_zoo_summary()

    # Verify the constraint: which models fit in 4 MB as INT8?
    print("── 4 MB deployment constraint check ──")
    for name in list_models():
        entry = get_model(name)
        fits = entry.size_int8_MB() <= 4.0
        print(f"  {entry.name:<20} INT8: {entry.size_int8_MB():.1f} MB "
              f"→ {'✓ fits' if fits else '✗ exceeds'} 4 MB budget")
    print()