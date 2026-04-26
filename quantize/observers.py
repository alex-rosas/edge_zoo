"""
quantize/observers.py
---------------------
Calibration DataLoader factory and observer inspection utilities.

This file supplies the two DataLoaders consumed by quantize/pipeline.py
Stage 3 (calibrate()). The two loaders are the only thing that differs
between the two conditions of the miscalibration experiment
(experiments/miscalibration.py):

  - real_calibration_loader()     CIFAR-10 resized to 224×224, ImageNet stats
  - gaussian_calibration_loader() Random N(0,1) tensors, same shape

Everything else in the pipeline — architecture, observer type, bit-width,
n_batches — is held constant across conditions. That single-variable design
makes the accuracy delta between conditions attributable to calibration
data quality alone.

Observer inspection:
  - print_observer_ranges()  Print [α, β] per layer after calibration
  - get_observer_ranges()    Return ranges as a dict for programmatic use
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets


# ─── ImageNet normalisation constants ────────────────────────────────────────
# Used by all torchvision pre-trained models. Applying these to calibration
# data is not optional: the models were trained with this normalisation, so
# their internal activation distributions assume it. Using un-normalised
# images would shift every observer range by a constant — a systematic error
# that would compound the miscalibration we are trying to isolate.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Real-image calibration loader ───────────────────────────────────────────

def real_calibration_loader(
    data_dir: str = "./data",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
) -> DataLoader:
    """
    CIFAR-10 resized to 224×224 with ImageNet normalisation.

    Why CIFAR-10 and not ImageNet:
      ImageNet requires a manual download and licence agreement. CIFAR-10
      downloads automatically via torchvision and is available in Colab
      without any setup. Resizing to 224×224 and applying ImageNet statistics
      gives the models activation distributions that are representative of
      natural images — which is the property we need for calibration, not
      domain-exact matching.

    Why n_batches * batch_size images:
      We only need enough images for the observers to accumulate stable
      range estimates. 100 batches × 32 images = 3200 images. This is
      sufficient for MinMaxObserver — it only needs to see the extremes of
      the activation distribution, not model it statistically.

    Args:
        data_dir:    Directory where CIFAR-10 will be downloaded.
        batch_size:  Images per batch. Default: 32.
        n_batches:   How many batches calibrate() will consume. Used here
                     only to size the dataset; the loader itself is infinite
                     from calibrate()'s perspective.
        image_size:  Resize target. Must match model input_shape. Default: 224.
        num_workers: DataLoader worker processes. Default: 2.

    Returns:
        A DataLoader yielding (images, labels) with ImageNet normalisation.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,       # use test split — no risk of leaking training data
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,      # shuffle so observers see varied classes per batch
        num_workers=num_workers,
        pin_memory=False,
    )

    print(
        f"  Real calibration loader: CIFAR-10 ({len(dataset)} images, "
        f"batch_size={batch_size}, image_size={image_size})"
    )
    return loader


# ─── Gaussian calibration loader ─────────────────────────────────────────────

class _GaussianDataset(Dataset):
    """
    Synthetic dataset of random N(0,1) tensors.

    Each item is an independent draw — there is no structure, no class
    signal, no spatial correlation. The activation ranges observers
    accumulate from this data will reflect the Gaussian prior, not the
    true distribution of natural image activations.

    This is the broken calibration condition. Its purpose is to produce
    a measurable accuracy degradation that isolates calibration data
    quality as the causal variable.
    """

    def __init__(self, n_samples: int, image_size: int):
        self.n_samples  = n_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Draw a fresh random tensor each time — no caching, maximum variance.
        x = torch.randn(3, self.image_size, self.image_size)
        return x, 0   # label 0 is a placeholder; calibrate() discards labels


def gaussian_calibration_loader(
    batch_size: int = 32,
    n_batches: int = 100,
    image_size: int = 224,
    num_workers: int = 0,
) -> DataLoader:
    """
    Random N(0,1) tensors, same shape as real calibration images.

    This is the miscalibration condition. The observers will record
    activation ranges driven by Gaussian inputs, which diverge from the
    true activation ranges of natural images — most severely in early
    layers, where edge detectors and colour filters produce strongly
    non-Gaussian activation distributions.

    Why N(0,1) specifically:
      Mean 0 and unit variance are the natural "uninformed prior" for a
      normalised input. It is the maximum-entropy distribution over the
      real line with fixed mean and variance — a principled choice of
      bad calibration data, not an arbitrary one.

    num_workers=0 default: random tensors are generated on-the-fly; there
    is no I/O to parallelise, so worker processes add overhead without benefit.

    Args:
        batch_size:  Images per batch. Default: 32.
        n_batches:   Determines dataset size (n_batches × batch_size).
        image_size:  Must match model input_shape. Default: 224.
        num_workers: DataLoader workers. Default: 0.

    Returns:
        A DataLoader yielding (gaussian_tensor, 0) pairs.
    """
    n_samples = n_batches * batch_size
    dataset   = _GaussianDataset(n_samples=n_samples, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,     # already random; shuffling adds cost with no benefit
        num_workers=num_workers,
    )

    print(
        f"  Gaussian calibration loader: N(0,1) ({n_samples} tensors, "
        f"batch_size={batch_size}, image_size={image_size})"
    )
    return loader


# ─── Observer inspection utilities ───────────────────────────────────────────

def get_observer_ranges(prepared_model: nn.Module) -> dict:
    ranges = {}
    for name, module in prepared_model.named_modules():
        if not (hasattr(module, "min_val") and hasattr(module, "max_val")):
            continue

        min_val = module.min_val
        max_val = module.max_val

        # Narrow the type from Tensor | Module | Unknown to Tensor.
        # Pylance cannot infer this from nn.Module's generic __getattr__.
        if not (isinstance(min_val, torch.Tensor) and isinstance(max_val, torch.Tensor)):
            continue

        # Skip observers that were never triggered.
        if min_val.numel() == 0 or torch.isinf(min_val).all():
            continue

        ranges[name] = {
            "min": float(min_val.min()),
            "max": float(max_val.max()),
        }

    return ranges

def print_observer_ranges(prepared_model: nn.Module) -> None:
    """
    Print a formatted table of observer ranges after calibration.

    Useful for:
      - Verifying calibration ran (all ranges should be finite)
      - Comparing real vs Gaussian calibration side by side
      - Identifying layers with unusually wide ranges (outlier sensitivity)

    Example output:
      Layer                              |    min |    max | width
      ─────────────────────────────────────────────────────────────
      activation_post_process_0         |  -2.14 |   3.87 |  6.01
      activation_post_process_1         |   0.00 |   4.23 |  4.23
      ...
    """
    ranges = get_observer_ranges(prepared_model)

    if not ranges:
        print("  No observer ranges found — was calibrate() called?")
        return

    header = f"  {'Layer':<40} | {'min':>7} | {'max':>7} | {'width':>7}"
    print(f"\n{header}")
    print(f"  {'─' * 40}-+-{'─' * 7}-+-{'─' * 7}-+-{'─' * 7}")

    for name, r in ranges.items():
        width = r["max"] - r["min"]
        short_name = name[-40:] if len(name) > 40 else name
        print(f"  {short_name:<40} | {r['min']:>7.3f} | {r['max']:>7.3f} | {width:>7.3f}")

    print(f"\n  Total observers populated: {len(ranges)}")

# ─── Accuracy evaluation ──────────────────────────────────────────────────────

def evaluate_top1(
    model: nn.Module,
    data_loader,
    device: str = "cpu",
    max_batches: int = 50,
) -> float:
    """
    Compute top-1 accuracy on a DataLoader subset.

    max_batches=50 with batch_size=32 gives 1600 images — sufficient for
    a stable accuracy estimate under PTQ conditions. Running on the full
    CIFAR-10 test set (10,000 images) would be more precise but adds
    runtime with diminishing returns for a comparative experiment where
    we care about the delta between conditions, not the absolute number.

    Note on CIFAR-10 vs ImageNet accuracy:
      The models were trained on ImageNet; CIFAR-10 classes overlap only
      partially with ImageNet classes. The absolute top-1 numbers will be
      lower than the published ImageNet figures. What matters here is the
      delta between real and Gaussian calibration on the same evaluation
      set — not the absolute value.

    Args:
        model:       Any nn.Module (FP32 or INT8).
        data_loader: DataLoader yielding (images, labels).
        device:      Compute device. Default: "cpu".
        max_batches: Batches to evaluate. Default: 50.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    model.eval()
    model.to(device)

    correct = 0
    total   = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs    = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)

    return correct / total if total > 0 else 0.0