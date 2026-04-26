"""
quantize/observers.py
---------------------
Calibration DataLoader factory and observer inspection utilities.

Two calibration loaders (for the miscalibration experiment):
  - real_calibration_loader()     CIFAR-10 resized to 224x224, ImageNet stats
  - gaussian_calibration_loader() Random N(0,1) tensors, same shape

Evaluation loader:
  - imagenet_eval_loader()        evanarlian/imagenet_1k_resized_256 via HF
                                  streaming. No licence or token required.
                                  Images are pre-resized to 256px; we apply
                                  CenterCrop(224) and ImageNet normalisation.
                                  Produces top-1 numbers directly comparable
                                  to published benchmarks.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as T
import torchvision.datasets as datasets


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Real-image calibration loader (CIFAR-10) ────────────────────────────────

def real_calibration_loader(
    data_dir: str = "./data",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
) -> DataLoader:
    """
    CIFAR-10 resized to 224x224 with ImageNet normalisation.
    Used for calibration only — not for accuracy evaluation.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(
        f"  Real calibration loader: CIFAR-10 ({len(dataset)} images, "
        f"batch_size={batch_size}, image_size={image_size})"
    )
    return loader


# ─── ImageNet evaluation loader (HF streaming, no token required) ─────────────

class _HFImageNetDataset(IterableDataset):
    """
    Wraps evanarlian/imagenet_1k_resized_256 validation split as a PyTorch
    IterableDataset using HF streaming. No licence or token required.

    Images are already resized to 256px shortest side. We apply CenterCrop(224)
    and ImageNet normalisation — the standard torchvision evaluation pipeline.
    """

    def __init__(self, max_samples: int = 2000):
        self.max_samples = max_samples
        self.transform = T.Compose([
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(
            "evanarlian/imagenet_1k_resized_256",
            split="val",
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        for sample in ds:
            if count >= self.max_samples:
                break
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = sample["label"]
            yield self.transform(img), label
            count += 1


def imagenet_eval_loader(
    batch_size: int = 32,
    max_samples: int = 2000,
) -> DataLoader:
    """
    ImageNet-1k validation set via HF streaming.
    No token or licence required — evanarlian/imagenet_1k_resized_256 is public.

    Args:
        batch_size:  Images per batch. Default: 32.
        max_samples: Validation images to use. Default: 2000.

    Returns:
        DataLoader yielding (images, labels) with ImageNet normalisation.
    """
    dataset = _HFImageNetDataset(max_samples=max_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,   # IterableDataset + streaming — no multiprocessing
    )

    print(
        f"  ImageNet eval loader: HF streaming evanarlian/imagenet_1k_resized_256 "
        f"(max {max_samples} images, batch_size={batch_size})"
    )
    return loader


# ─── Gaussian calibration loader ─────────────────────────────────────────────

class _GaussianDataset(Dataset):
    def __init__(self, n_samples: int, image_size: int):
        self.n_samples  = n_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return torch.randn(3, self.image_size, self.image_size), 0


def gaussian_calibration_loader(
    batch_size: int = 32,
    n_batches: int = 100,
    image_size: int = 224,
    num_workers: int = 0,
) -> DataLoader:
    """Random N(0,1) tensors — the miscalibration condition."""
    n_samples = n_batches * batch_size
    dataset   = _GaussianDataset(n_samples=n_samples, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
        if not (isinstance(min_val, torch.Tensor) and isinstance(max_val, torch.Tensor)):
            continue
        if min_val.numel() == 0 or torch.isinf(min_val).all():
            continue
        ranges[name] = {
            "min": float(min_val.min()),
            "max": float(max_val.max()),
        }
    return ranges


def print_observer_ranges(prepared_model: nn.Module) -> None:
    ranges = get_observer_ranges(prepared_model)
    if not ranges:
        print("  No observer ranges found — was calibrate() called?")
        return
    header = f"  {'Layer':<40} | {'min':>7} | {'max':>7} | {'width':>7}"
    print(f"\n{header}")
    print(f"  {'─'*40}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")
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

    Args:
        model:       Any nn.Module (FP32 or INT8).
        data_loader: DataLoader yielding (images, labels).
        device:      Compute device.
        max_batches: Batches to evaluate.

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
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0