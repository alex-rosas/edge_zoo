# EdgeZoo

**A CNN quantization diagnostic pipeline for edge AI deployment.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![Models](https://img.shields.io/badge/models-3-green)
![Eval](https://img.shields.io/badge/ImageNet%20top--1%20delta-0.25%25%20%7C%207.19%25%20%7C%2038%25-red)

> EdgeZoo is a **diagnostic pipeline** — not a deployment tool. It runs PTQ on three CNN architectures, attributes quantization error layer-by-layer, and produces a controlled observer comparison to explain which accuracy losses are recoverable and which are architectural.

---

## What Is EdgeZoo?

EdgeZoo applies **post-training quantization (PTQ)** to three CNN families — residual, depthwise-separable, and compound-scaled — and measures the cost at every level: accuracy, latency, size, ONNX operator coverage, and per-layer L2 error.

It is **not** a QAT trainer. It is **not** an NPU deployment tool. It is a pipeline that tells you, before you commit to a model, where quantization costs are concentrated and whether the observer choice matters.

```
FP32 model
    │
    ▼
[Stage 1] fold_batchnorm()       deep-copy + eval mode
    │
    ▼
[Stage 2] insert_observers()     FX graph capture + observer insertion
    │
    ▼
[Stage 3] calibrate()            100 forward passes on CIFAR-10
    │
    ▼
[Stage 4] convert_to_int8()      freeze ranges → lower to INT8 ops
    │
    ▼
[Stage 5] export_onnx()          opset 17, dynamo=False
    │
    ▼
INT8 ONNX model  +  accuracy delta  +  layer L2 error table
```

---

## Results

| Model | FP32 top-1 | INT8 top-1 | Δ Acc | INT8 MB | Fits 4 MB | ONNX coverage |
|---|---|---|---|---|---|---|
| ResNet-18 | 67.81% | 67.56% | −0.25% | 11.26 | ✗ | 62.7% |
| MobileNetV2 | 67.31% | 60.12% | −7.19% | 3.67 | ✓ | 67.7% |
| EfficientNet-B0 | 74.45% | 36.46% | −38.00% | 5.63 | ✗ | 64.8% |

*Evaluation: strided ImageNet val — every 25th image, 2 per class, 2016 total, fully deterministic.*
*FP32 latency measured on CUDA (T4). INT8 latency measured on CPU (quantization is CPU-only).*

### Observer Comparison

| Model | Δ fbgemm (HistogramObs) | Δ minmax (MinMaxObs) | Observer sensitive? |
|---|---|---|---|
| ResNet-18 | 0.25% | 2.88% | No |
| MobileNetV2 | 7.19% | 67.21% (collapse) | Critical |
| EfficientNet-B0 | 38.00% | 21.83% | Yes — worse under both |

### Hypothesis Outcomes

| Hypothesis | Outcome |
|---|---|
| H1: MobileNetV2 depthwise layers rank highest in L2 error | Partially confirmed |
| H2: ~4× size reduction, 1.5–2.5× latency improvement | Confirmed (size); latency complicated by CPU/CUDA split |
| H3: Real calibration outperforms Gaussian ≥ 0.5% | Pending |
| H4: INT8 speedup grows with matrix size | **Refuted** on both Apple Silicon and x86 |

---

## How to Run

### Prerequisites

```bash
git clone https://github.com/alex-rosas/edge_zoo.git
cd edge_zoo
pip install torch torchvision datasets onnx onnxscript pandas
```

### Main benchmark (fbgemm observer, all 3 models)

```bash
PYTHONPATH=/path/to/edge_zoo python benchmark/bench.py
```

### Observer comparison experiment

```bash
PYTHONPATH=/path/to/edge_zoo python experiments/observer_comparison.py
```

### C++ matrix benchmark (H4)

```bash
cd bench_cpp
cmake -B build && cmake --build build
./build/bench_matmul
```

### Colab one-liner

```python
!git clone https://github.com/alex-rosas/edge_zoo.git
%cd /content/edge_zoo
!pip install datasets onnxscript -q
!PYTHONPATH=/content/edge_zoo python benchmark/bench.py
```

### Environment

| Variable | Default | Description |
|---|---|---|
| `PYTHONPATH` | — | Must include repo root |
| `DATA_DIR` | `./data` | CIFAR-10 download location |
| `HF_TOKEN` | — | Optional — raises HF Hub rate limits |

---

## Project Structure

```
edge_zoo/
├── models/
│   └── zoo.py                 # Model registry — ModelEntry dataclass, get_zoo(), print_zoo_summary()
├── quantize/
│   ├── pipeline.py            # Five-stage PTQ pipeline — run_pipeline(entry, cal_loader, qconfig=)
│   ├── observers.py           # Data loaders — imagenet_eval_loader(), real_calibration_loader()
│   └── error_analysis.py      # Layer-wise L2 attribution — forward hooks, dequantize() handling
├── benchmark/
│   └── bench.py               # Main benchmark — runs all 3 models, prints tradeoff table
├── experiments/
│   ├── observer_comparison.py # fbgemm vs minmax — controlled experiment, comparison table
│   └── miscalibration.py      # Real vs Gaussian calibration — H3 experiment (pending)
├── bench_cpp/
│   ├── bench.cpp              # INT8 vs FP32 C++ matmul benchmark — H4 experiment
│   └── CMakeLists.txt
├── configs/
│   └── recipe.yaml            # Deployment recipe for MobileNetV2 INT8
├── onnx_models/               # Exported .onnx files (generated)
├── results/                   # CSVs — tradeoff.csv, layer_errors.csv, onnx_graph.csv (generated)
└── docs/
    └── case_study.md          # Engineering decisions, failure forensics, what was learned
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| PTQ pipeline | PyTorch FX graph mode (`prepare_fx` / `convert_fx`) | Automatic BN-fold, graph-level observer insertion |
| Observer | `get_default_qconfig_mapping("fbgemm")` | HistogramObserver — outlier-robust; MinMaxObserver causes collapse on MobileNetV2 |
| Calibration data | CIFAR-10 (torchvision auto-download) | Auto-downloads; controlled variable isolation for observer comparison |
| Evaluation data | HF `evanarlian/imagenet_1k_resized_256`, strided | Correct labels, deterministic 2-per-class coverage, verified 50 images/class |
| ONNX export | PyTorch legacy exporter, opset 17, `dynamo=False` | Dynamo exporter crashes on `Conv2dPackedParamsBase` |
| C++ benchmark | Standard C++17, `<chrono>`, `<cstdint>` | Measures compiler output, not hand-optimised kernel |
| Results | pandas CSV | Reproducible, diffable |

---

## Key Engineering Decisions

**PTQ over QAT.** No training pipeline access, one-day constraint. QAT achieves higher accuracy but requires training epochs. Full reasoning in `docs/case_study.md`.

**HistogramObserver as default.** MinMaxObserver causes quantization collapse on MobileNetV2 — all inputs predicted as class 739. The observer comparison experiment quantifies the difference. Full forensic in `docs/case_study.md`.

**`dynamo=False` in ONNX export.** PyTorch 2.x's dynamo exporter crashes on quantized models with `Conv2dPackedParamsBase`. The legacy exporter handles these correctly. Full account in `docs/case_study.md`.

**Strided ImageNet evaluation.** The HF ImageNet dataset is sorted by class; streaming shuffle only covers ~4 classes per buffer window. Strided sampling (every 25th image) gives deterministic, class-balanced coverage. Verified: exactly 50 images per class. Full forensic in `docs/case_study.md`.

**Standard C++ only in the H4 benchmark.** Using NEON SDOT explicitly would confirm H4 rather than test it. The absence of explicit SIMD is the finding — INT8 speedup requires hand-optimised kernels, not just a different dtype. Full account in `docs/case_study.md`.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| EfficientNet-B0 loses 38% accuracy under PTQ | Not deployable without QAT or SE-block-specific configuration |
| Calibration uses CIFAR-10, not ImageNet | Uncontrolled variable; may inflate MobileNetV2 delta |
| INT8 latency measured on CPU, FP32 on CUDA | Cross-device comparison is not a fair latency benchmark |
| H3 (miscalibration experiment) not run end-to-end | Gaussian vs real calibration finding is unconfirmed |
| HF dataset downloads all splits (~50 GB) | Requires substantial Colab disk space |

*Root causes and fixes in `docs/case_study.md`.*

---

## What Was Intentionally Not Built

| Item | Reason |
|---|---|
| QAT | No training pipeline; out of scope for a PTQ diagnostic |
| NPU deployment | Hardware-specific; out of scope for a portable diagnostic tool |
| MLIR / TVM | Separate toolchain; would merge compiler and quantization experiments |
| SIMD intrinsics in bench.cpp | Their absence is the finding; adding them would change the experiment |
| ImageNet calibration | CIFAR-10 isolates calibration domain as a controlled variable |
| Miscalibration end-to-end run | Infrastructure exists in `experiments/miscalibration.py`; pending execution |

---

## Known Technical Debt

1. **Calibration loader has no fixed seed.** `shuffle=True` with no `generator` produces different batch orderings per run, shifting HistogramObserver ranges slightly. Fix: `torch.Generator().manual_seed(42)` in `real_calibration_loader()`.
2. **HF dataset downloads all splits.** Only val is needed. Fix: pass `split="val"` only; cache to a known directory.
3. **`onnxscript` not in `colab_setup.py`.** Must be installed manually before running bench. Fix: add to setup script.
4. **ONNX filenames differ between bench.py and observer_comparison.py.** bench.py writes `model_int8.onnx`; comparison writes `model_fbgemm_int8.onnx`. Fix: unify naming convention.
5. **H3 experiment not run.** `experiments/miscalibration.py` exists and is instrumented; it has not been run end-to-end with the fixed evaluation dataset.

---

## Phases

- [x] Phase 0 — Problem definition, four hypotheses, model zoo
- [x] Phase 1 — PTQ pipeline (5 stages), FX graph mode, CIFAR-10 calibration
- [x] Phase 2 — Accuracy evaluation, layer-wise L2 attribution, ONNX interrogation
- [x] Phase 3 — C++ INT8/FP32 benchmark, H4 refuted on Apple Silicon and x86
- [x] Phase 4 — Observer comparison experiment, `pipeline.py` parameterised on `qconfig`
- [ ] Phase 5 — Miscalibration experiment (H3), SIMD benchmark extension

---

## Case Study

[docs/case_study.md](docs/case_study.md) — Engineering decisions, failure forensics, and what the project found.

Covers: the S_C clipping bug and two wrong hypotheses, the dynamo ONNX crash, the MinMaxObserver collapse forensic, the sorted-dataset problem, observer sensitivity by architecture, and what each of these implies for the next system.
