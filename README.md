# EdgeZoo

**A CNN quantization diagnostic pipeline for edge AI deployment.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.11.0-orange)
![C++](https://img.shields.io/badge/c%2B%2B-17-lightgrey)
![Phases](https://img.shields.io/badge/phases-4%2F4-green)

EdgeZoo is a **diagnostic pipeline**, not a deployment tool. It runs
post-training quantization across three CNN architectures and instruments
every stage to find where quantization fails — not just whether it works.

> NOT a model optimisation framework. NOT a benchmark suite.
> A controlled experiment designed to produce specific, falsifiable findings.

---

## Try It

### Python pipeline (Colab)

```bash
git clone https://github.com/USERNAME/edge_zoo.git
cd edge_zoo
python colab_setup.py
python benchmark/bench.py
```

### C++ benchmark (Mac / Linux)

```bash
cd bench_cpp
clang++ -std=c++17 -O2 -march=native -o bench bench.cpp
./bench
```

---

## What Is EdgeZoo?

EdgeZoo is a **five-stage PTQ pipeline with three diagnostic instruments**
attached, run across three CNN architectures simultaneously. Each diagnostic
instrument answers one question: where does calibration matter, which layers
absorb the most error, and what operators resist quantization.

It is not an ensemble system. It is not a hyperparameter search. The three
architectures are run in parallel because the architectural family
(`residual`, `depthwise-separable`, `compound-scaled`) is the explanatory
variable for the sensitivity hypothesis — not because more models produce
a better result.

```
torchvision pretrained weights
        │
        ▼
┌───────────────────┐
│   Model Registry  │  Uniform interface across 3 architectures
│   (models/zoo.py) │  Output: ModelEntry (weights, metadata, input_shape)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   PTQ Pipeline    │  BN-fold → observe → calibrate → INT8 → ONNX
│ (quantize/)       │  Input: ModelEntry + calibration DataLoader
└────────┬──────────┘  Output: fp32_model, int8_model, onnx_path
         │
         ├──────────────────────────┐
         ▼                          ▼
┌──────────────────┐    ┌───────────────────────┐
│  Layer-wise      │    │  ONNX Interrogator     │
│  Error Analysis  │    │  (onnx_interrogate.py) │
│ (error_analysis) │    │  Output: operator      │
│ Output: L2/max   │    │  coverage table        │
│ error per layer  │    └───────────────────────┘
└──────────────────┘
         │
         ▼
┌───────────────────┐
│  Miscalibration   │  Controlled experiment: real vs Gaussian calibration
│  Experiment       │  Single free variable — everything else held constant
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Result Tables    │  tradeoff.csv, layer_errors.csv, onnx_graph.csv
│  + YAML Recipe    │  Deployment recommendation tied to 4 MB / 10 ms
└───────────────────┘
```

---

## What It Produces

- **A deployment decision** — which of the three models fits the 4 MB INT8
  budget and survives quantization with acceptable accuracy loss
- **A sensitivity ranking** — which layers absorb the most quantization error,
  and whether that ranking is architecture-dependent
- **A calibration quality measurement** — how much accuracy is lost when
  calibration data does not match the inference distribution

---

## Demonstration — Before and After

**Governing question:** Given three CNNs with different accuracy and efficiency
profiles, which should be deployed under a 4 MB memory budget and 10 ms
latency constraint — and where does INT8 quantization cost most?

**Phase 1 answer (from metadata alone, before any experiment):**

| Model | FP32 size | INT8 size | Fits 4 MB? |
|---|---|---|---|
| ResNet-18 | 46.8 MB | 11.7 MB | ✗ |
| MobileNetV2 | 13.6 MB | 3.4 MB | ✓ |
| EfficientNet-B0 | 21.2 MB | 5.3 MB | ✗ |

**Phase 2–3 answer (after experiments):**

| Dimension | Baseline (FP32) | After INT8 PTQ |
|---|---|---|
| Model size | 13.6 MB | 3.4 MB (4.0× reduction) |
| Top-1 accuracy | 71.9% | [fill after run] |
| Accuracy delta | — | [fill after run] |
| Calibration sensitivity | — | [fill after run] |
| INT8 speedup (x86, 128×128) | 1.0× | 1.19× |
| INT8 speedup (x86, 1024×1024) | 1.0× | 0.43× |
| INT8 speedup (Apple M-series) | 1.0× | 0.13–0.20× |

---

## Results

### C++ scaling benchmark

| Size | Reps | FP32 (ms) | INT8 (ms) | Speedup |
|---|---|---|---|---|
| 128×128 | 1000 | 0.143 | 0.717 | 0.20 |
| 512×512 | 100 | 11.773 | 90.206 | 0.13 |
| 1024×1024 | 10 | 96.316 | 734.556 | 0.13 |

*Apple Silicon, Apple Clang, `-O2 -march=native`. See case study for x86 results.*

**H4 refuted:** INT8 does not outperform FP32 from compiled C++ on either
platform at scale. The advantage requires explicit SIMD intrinsics or a
dedicated integer pipeline. Full forensic in [docs/case_study.md](docs/case_study.md).

### Phase progression

| Phase | What was added | Key finding |
|---|---|---|
| 1 — Model Zoo | Registry of 3 CNNs with uniform interface | MobileNetV2 is the only model fitting 4 MB from metadata alone |
| 2 — PTQ Pipeline | 5-stage pipeline + 3 diagnostic instruments | [fill after run] |
| 3 — C++ Benchmark | INT8 vs FP32 matmul at 3 sizes | H4 refuted — compiler flag insufficient for INT8 SIMD advantage |
| 4 — Report | Deployment recipe + README | MobileNetV2 recommended under stated constraints |

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Model loading | `torchvision` | Pretrained weights, standard interface |
| PTQ API | `torch.ao.quantization` | FX graph mode — every stage auditable |
| ONNX export | `torch.onnx` + `onnxruntime` | Opset 17 — the format hardware compilers consume |
| Calibration data | CIFAR-10 (resized 224×224) | Auto-downloads, sufficient natural image statistics |
| Result tables | `pandas` | CSV output for `tradeoff.csv`, `layer_errors.csv`, `onnx_graph.csv` |
| C++ benchmark | Standard C++17 only | No BLAS — forces direct encounter with INT8 arithmetic |
| Build | CMake | Standard in embedded and automotive engineering |
| Runtime | Google Colab T4 | Python phases; C++ phases on local Mac |

---

## How to Run

### Prerequisites

- Python 3.11, PyTorch 2.11.0 (Colab T4 runtime provides both)
- `clang++` or `g++` with C++17 support (for C++ benchmark)
- CMake 3.16+ (optional — direct `clang++` command works without it)

### Environment variables

| Name | Default | Description |
|---|---|---|
| `EDGEZOO_DATA_DIR` | `./data` | Where CIFAR-10 is downloaded |
| `EDGEZOO_RESULTS_DIR` | `./results` | Where CSV result tables are written |

### Python pipeline (Colab)

```bash
# Cell 1 — first session only
!git clone https://github.com/USERNAME/edge_zoo.git
%cd edge_zoo

# Cell 2 — every session
!git pull
%run colab_setup.py

# Cell 3 — run full pipeline and produce result tables
!python benchmark/bench.py
```

### Miscalibration experiment only

```bash
python experiments/miscalibration.py
```

### C++ benchmark

```bash
cd bench_cpp
clang++ -std=c++17 -O2 -march=native -o bench bench.cpp
./bench

# Or with CMake
mkdir build && cd build
cmake .. && cmake --build .
./bench
```

---

## Key Engineering Decisions

**PTQ over QAT.** No training pipeline access, one-day constraint. QAT achieves
higher accuracy but requires training epochs and a full labelled dataset — neither
was available. Full reasoning in [docs/case_study.md](docs/case_study.md).

**MinMaxObserver as baseline.** Simple, deterministic, no hyperparameters. Its
sensitivity to outliers is a feature of the miscalibration experiment, not a
defect. Percentile and KL-divergence observers are a deferred next step.

**Per-channel weights, per-tensor activations.** Weight distributions vary
across output channels — per-channel is critical for weight accuracy.
Activations are more uniform within a layer — per-tensor avoids per-channel
scale vector overhead at inference.

**Standard C++ only in the benchmark.** No BLAS, no SIMD intrinsics. The
benchmark measures compiler output, not hand-optimised kernel output. That
distinction is the finding.

**Dynamic output scale in the C++ benchmark.** `S_C` is computed from the
actual FP32 output range per run. A fixed `S_C` clips nearly all outputs at
larger matrix sizes and produces meaningless error measurements.

Full reasoning for all decisions in [docs/case_study.md](docs/case_study.md).

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Accuracy evaluated on CIFAR-10, not ImageNet | Absolute numbers are lower than published benchmarks; deltas remain valid |
| MinMaxObserver sensitive to outliers | Small calibration sets may produce stretched grids; percentile observer would help |
| No real NPU latency | All timing is CPU wall-clock; INT8 NPU advantage is argued from arithmetic, not measured |
| INT8 speedup requires explicit SIMD | Compiler flag alone insufficient; no auto-vectorised INT8 advantage at large matrix sizes |

Root causes and concrete fixes in [docs/case_study.md](docs/case_study.md).

---

## What Was Intentionally Not Built

| Item | Reason |
|---|---|
| Quantization-aware training (QAT) | No training pipeline access; one-day constraint |
| Pruning / knowledge distillation | Each deserves a separate project; combining produces shallow treatment of all three |
| Real NPU deployment | Hardware not available |
| Percentile / KL-divergence observers | MinMaxObserver is the controlled baseline; alternatives are a deferred next step |
| Custom compiler passes (MLIR, TVM) | Out of scope; ONNX opset 17 is the interchange format this pipeline targets |
| Explicit SIMD intrinsics in C++ benchmark | Their absence is the finding — see H4 result |

---

## Known Technical Debt

1. **Null fields in `recipe.yaml`** — accuracy delta, layer error rankings, and
   ONNX coverage table are unfilled pending a full `benchmark/bench.py` run in Colab.
2. **CIFAR-10 calibration** — replacing with an ImageNet validation subset would
   make the accuracy numbers directly comparable to published benchmarks.
3. **Fixed rep counts in bench.cpp** — the 1000/100/10 schedule was chosen for
   Apple Silicon; on slower hardware the 512×512 and 1024×1024 runs may still
   be impractically slow.
4. **No reproducibility seed for calibration** — `observers.py` does not set a
   global seed before calibration, so results may vary slightly across runs with
   different PRNG state.

---

## Project Structure

```
edge_zoo/
│
├── models/
│   └── zoo.py                  Model registry — uniform interface for 3 CNNs
│
├── quantize/
│   ├── pipeline.py             Five-stage PTQ pipeline (BN-fold → ONNX export)
│   ├── observers.py            Calibration loaders, observer range inspection
│   ├── error_analysis.py       Layer-wise L2 and max-error attribution
│   └── onnx_interrogate.py     Quantized vs FP32 operator classification
│
├── experiments/
│   └── miscalibration.py       Controlled experiment — real vs Gaussian calibration
│
├── benchmark/
│   └── bench.py                Orchestration — runs pipeline, writes result tables
│
├── bench_cpp/
│   ├── bench.cpp               INT8 vs FP32 matmul at 3 sizes, no dependencies
│   └── CMakeLists.txt          CMake build: -O2 -march=native -std=c++17
│
├── configs/
│   └── recipe.yaml             Deployment recipe — MobileNetV2 INT8 parameters
│
├── results/
│   ├── tradeoff.csv            Accuracy delta, size, latency per model
│   ├── layer_errors.csv        Layer-wise error rankings per model
│   └── onnx_graph.csv          Quantized vs FP32 operator counts per model
│
├── docs/
│   └── case_study.md           Engineering decisions, failure forensics, lessons
│
├── colab_setup.py              One-command Colab session initialisation
└── README.md                   This file
```

---

## Phases

- [x] Phase 1 — Model Zoo (`models/zoo.py`)
- [x] Phase 2 — PTQ Pipeline + Diagnostics (`quantize/`, `experiments/`, `benchmark/`)
- [x] Phase 3 — C++ Scaling Benchmark (`bench_cpp/`)
- [x] Phase 4 — Deployment Recipe + Documentation (`configs/`, `docs/`)

---

## Case Study

[docs/case_study.md](docs/case_study.md) — Engineering decisions, failure
forensics, and project positioning.

Covers: the S_C clipping bug and how it was diagnosed; why H4 was stated on
the basis of Jacob et al. (2018) and what the experiment revealed about the
missing implementation condition; and what three things would be done
differently if the project were resumed.

---

## Author

Luis Alejandro Rosas Martínez — PhD in Applied Mathematics, Institut
Polytechnique de Paris