# EdgeZoo — Case Study

**Engineering decisions, failure forensics, and what the project actually found.**

---

## What Makes This Hard

Post-training quantization is not a function call. Five sequential pipeline stages — each a precondition for the next — interact in ways that are invisible without instrumentation. Getting one stage wrong produces a silently incorrect model, not an error. Three specific hard problems defined this project:

**Calibration data quality is invisible until you instrument it.** A badly calibrated model produces INT8 outputs that look plausible — the accuracy drop is the only signal. Without a controlled experiment that isolates calibration data as the single free variable, the degradation has no explanation.

**Quantization error does not distribute uniformly across layers.** A global accuracy delta tells you PTQ costs something. It does not tell you where. The forward hook infrastructure for layer-wise attribution was built specifically to make that invisible distribution visible.

**Observer choice is architecture-dependent in ways that are not obvious from the literature.** MinMaxObserver is the standard baseline in PTQ tutorials. This project discovered that it causes complete quantization collapse on depthwise-separable architectures — a finding that is not documented in PyTorch's own quantization guides.

---

## When to Use This (vs Alternatives)

✅ **Use this when:**
- You have a trained model, no training pipeline access, and a hard memory budget. PTQ under a one-day constraint with no labelled data is the realistic scenario this project proxies.
- You need to understand *where* quantization costs before committing to a model for hardware deployment.
- You are handing a model to a hardware compiler team and need to document ONNX operator coverage gaps before integration.

❌ **Don't use this when:**

| Scenario | Why | What to use instead |
|---|---|---|
| Accuracy loss > 1% is unacceptable | PTQ has no recovery mechanism | QAT with full training pipeline |
| Target is a Transformer or LLM | Attention patterns and activation distributions require specialised treatment | GPTQ, AWQ, or SmoothQuant |
| You need real NPU latency | All timing here is CPU wall-clock | On-device profiling with NXP toolchain |
| EfficientNet-B0 is the target model | Its Sigmoid+Mul squeeze-and-excitation blocks resist quantization under both observers tested | Model-specific PTQ configuration or QAT |

---

## Industry Context and Positioning

| Alternative | What it does | Gap vs EdgeZoo |
|---|---|---|
| PyTorch quantization tutorial | Demonstrates PTQ on a single model | No failure analysis, no controlled experiment, no architectural comparison |
| Hugging Face Optimum | Production INT8 export for Transformers | CNN-focused, no layer-wise diagnostic, no observer comparison |
| ONNX Runtime quantization tools | Graph-level quantization via onnxruntime | No calibration quality experiment, no operator coverage analysis |
| TensorFlow Lite converter | End-to-end mobile deployment pipeline | TF ecosystem only, no PyTorch FX graph mode |
| gemmlowp / XNNPACK | Hand-optimised integer kernel libraries | Library, not a diagnostic pipeline |

EdgeZoo is not a deployment tool. It is a diagnostic pipeline. A deployment tool optimises for a metric; a diagnostic pipeline produces findings that explain which metric is achievable and why.

---

## Technical Approach

### The data contract

Every downstream function receives a `ModelEntry` from the registry rather than loading a model directly. This means the pipeline, the benchmarks, and the error attribution all operate on a common interface regardless of architecture. The registry stores `arch_family` as an explicit field because the architectural label is the explanatory variable for H1 — without it, the attribution results are numbers without a frame.

### The PTQ pipeline as a sequencing constraint

The five stages are not a design choice — they are a constraint imposed by PyTorch's FX graph mode. BN-folding must precede observer insertion to prevent double-scaling. Calibration must precede INT8 conversion because `convert_fx()` reads observer ranges. ONNX export must follow conversion because INT8 operators only exist in the converted graph. Stage 2 deep-copies the model before any graph mutation, preserving the original FP32 weights for layer-wise attribution — without this, the reference signal is destroyed.

### The evaluation design

The evaluation uses strided sampling from `evanarlian/imagenet_1k_resized_256`: every 25th image from the 50,000-image validation set, giving exactly 2 images per class across all 1000 classes. This was verified empirically — the dataset has exactly 50 images per class in strict sequential order. Strided sampling gives uniform class coverage with zero randomness, making the FP32/INT8 accuracy delta fully deterministic and reproducible across sessions. All models evaluate on the same preloaded batch list; streaming loaders would produce different samples per evaluation call.

### The observer capture window

Observer ranges — the `[α, β]` per-layer min/max values — exist only between Stage 3 (calibration) and Stage 4 (conversion). After `convert_fx()`, observer nodes are replaced with static scale and zero-point constants. The miscalibration experiment must capture ranges in this window or the comparison is impossible to reconstruct.

---

## Key Engineering Decisions

### PTQ over QAT

**Chosen:** Post-training quantization with no retraining.
**Rejected:** Quantization-aware training.
**Why:** No access to the training pipeline or labelled training data; one-day constraint. QAT achieves higher final accuracy by exposing the model to quantization noise during training, but it requires training epochs and a full dataset. PTQ operates on a frozen model. Under the stated constraints, PTQ is the only viable option — framing it as a simplification would prevent the conversation about when QAT is worth its cost.

### fbgemm default over MinMaxObserver

**Chosen:** `get_default_qconfig_mapping("fbgemm")` — HistogramObserver for activations, percentile-based outlier clipping.
**Rejected:** Custom `MinMaxObserver` QConfig — deterministic, no hyperparameters, tracks absolute min/max.
**Why:** MinMaxObserver was the original choice and caused quantization collapse on MobileNetV2 and EfficientNet-B0 under CIFAR-10 calibration. The INT8 model predicted a single class (739 — parking meter) for all inputs. `HistogramObserver` clips outliers at the 99.99th percentile, producing stable grids. The switch was validated by the observer comparison experiment. MinMaxObserver is retained in `experiments/miscalibration.py` where its sensitivity to outliers is the controlled variable.

### dynamo=False in ONNX export

**Chosen:** Legacy ONNX exporter (`dynamo=False`).
**Rejected:** Default dynamo-based exporter (PyTorch 2.x default).
**Why:** The dynamo exporter traces models using `torch.export.export()`, which attempts to flatten all custom C++ objects including `Conv2dPackedParamsBase` — the internal representation of quantized conv layers. This flattening is not implemented for quantized models. The crash was discovered during the first full benchmark run and was not documented in PyTorch's migration guide. `dynamo=False` is a forward-compatibility guard for any pipeline that exports quantized models.

### Strided ImageNet evaluation

**Chosen:** Every 25th image from the sorted validation set — 2 images per class, 2016 total.
**Rejected:** Random sampling with shuffle; streaming with buffer shuffle.
**Why:** The dataset is sorted by class with exactly 50 images per class. Streaming shuffle is ineffective because the dataset is partitioned into 52 shards each containing sequential per-class data. Buffer-size shuffling only covers ~4 classes at a time. Random sampling with a seed produced different images across Colab sessions due to non-deterministic HF cache behaviour. Strided sampling is deterministic, class-balanced, and verified to cover all 1000 classes regardless of session state.

### Standard C++ only in the benchmark

**Chosen:** No external libraries — only `<chrono>` and `<cstdint>`.
**Rejected:** OpenBLAS, explicit NEON intrinsics, BLAS Accelerate.
**Why:** The benchmark is designed to measure what a compiler produces from a readable implementation, not what a hand-optimised kernel achieves. Using NEON SDOT explicitly would confirm H4 rather than test it. The absence of explicit SIMD is the finding — it reveals that the INT8 arithmetic advantage requires explicit implementation, not just a compiler flag. This distinction directly motivates why NPUs exist.

### Dynamic S_C in the C++ benchmark

**Chosen:** `S_C = max(|C_fp32|) / 127` computed from the actual reference output per run.
**Rejected:** Fixed `S_C = 0.01`.
**Why:** The output magnitude of a matrix multiply scales with K. At K=128, values reach ±300; at K=1024, ±2400. A fixed S_C clips nearly every output element at larger sizes, and the error measurement reports clipping error rather than quantization error. This was a bug discovered during the first benchmark run and is documented below.

---

## The S_C Bug — A Forensic Account

**Phase:** 3 — C++ benchmark, first run on Apple Silicon.

**Symptom:**
```
Size    FP32 (ms)  INT8 (ms)  Speedup  Max Abs Error
128×128  0.155      0.717      0.217    93.466
512×512  12.653     107.712    0.117    215.563
[benchmark hanging at 1024×1024]
```

**Wrong hypothesis 1 — formatting issue.** The error of 93 looked like a unit mismatch or a scale factor applied twice. Inspected `max_abs_error()` — the dequantization was `C_i8[i] * S_C`, which is correct. Ruled out.

**Wrong hypothesis 2 — INT8 output was all zeros.** If the quantized matmul produced zeros, the error would equal `max(|C_fp32|)`. Printed a slice of `C_i8`. The outputs were ±127 almost everywhere — saturation, not silence.

**Root cause.** `S_C = 0.01` mapped the output grid to `[-1.28, 1.27]`. The actual output of a 128×128 matmul with inputs in `[-2.54, 2.54]` reaches approximately ±300. The requantization step multiplied every INT32 accumulator by `S_A * S_B / S_C = 0.04`, producing scaled values in the thousands — all clipped to ±127. The error of 93 was measuring clipping, not quantization. The hang was a separate issue: 1000 repetitions at 1024×1024 takes ~96 seconds on Apple Silicon.

**Fix.** Compute `S_C = max(|C_fp32|) / 127` from the reference output before each run. Scale reps: 1000/100/10 by matrix size.

**What would have found it faster.** Printing one element of `C_i8` before the full benchmark would have shown saturation immediately.

---

## The dynamo ONNX Export Crash — A Forensic Account

**Phase:** 2 — first full `benchmark/bench.py` run in Colab.

**Symptom:**
```
AttributeError: __torch__.torch.classes.quantized.Conv2dPackedParamsBase
does not have a field with name '__obj_flatten__'
torch.onnx._internal.exporter._errors.TorchExportError
```

**Root cause.** PyTorch 2.x changed the default ONNX exporter to a dynamo-based backend that uses `torch.export.export()`. This attempts to flatten all custom C++ objects, including `Conv2dPackedParamsBase` — the internal representation of quantized conv layers. Flattening is not implemented for this type.

**Fix.** `dynamo=False` in `torch.onnx.export()`. One line, in the right place.

**What to remember.** Any PyTorch version update may change the default ONNX exporter. `dynamo=False` is a required forward-compatibility guard for quantized model export.

---

## The MinMaxObserver Collapse — A Forensic Account

**Phase:** 2 — accuracy evaluation, first run with ImageNet labels.

**Symptom:** MobileNetV2 INT8 accuracy: 0.00%. Every input predicted class 739 (parking meter). INT8 output logit for class 739: 20.46 (vs FP32: 2.36).

**Diagnosis path.** The output logits had reasonable range and std — not a classic collapse. Class 739 just had a systematically inflated logit. Isolated the issue to `features.18.2` (last feature layer before classifier): FP32 features std 0.97, INT8 features std 1.76, L2 error 1021. The quantization was amplifying activations, not approximating them. The final Linear layer received a corrupted signal.

**Root cause.** MinMaxObserver tracks the absolute min and max activation value seen during calibration. CIFAR-10 images, resized to 224×224, drive activations to ranges that diverge substantially from ImageNet activations in early layers. One or more extreme outlier values stretched the INT8 grid across a range that never appears at inference. The 256 steps are distributed across mostly-empty space, with most real activations mapping to the same INT8 bucket.

**Fix.** Replace `MinMaxObserver` with `get_default_qconfig_mapping("fbgemm")`, which uses `HistogramObserver` — percentile-based, clips at 99.99th percentile. One-line change in `pipeline.py`.

**Architectural implication.** Observer choice is architecture-dependent. ResNet-18 (standard Conv, residual) was robust to both observers. MobileNetV2 (depthwise-separable) collapsed completely. EfficientNet-B0 (Sigmoid+Mul squeeze-and-excitation) was badly degraded under both. The observer comparison experiment quantifies this precisely.

---

## The Sorted Dataset Problem — A Forensic Account

**Phase:** 2 — accuracy evaluation, after switching to ImageNet.

**Symptom:** All labels in first 3 batches (96 images) were class 0 or 1. MobileNetV2 FP32 accuracy: 67.56%. MobileNetV2 INT8 accuracy: 0.00% — delta 67.56%.

**Wrong hypothesis 1 — shuffle not applied.** Added `shuffle=True` to the streaming loader. Still 2 unique classes in 96 images. The HF streaming shuffle uses a buffer that only covers ~4 classes at a time because each shard contains sequential per-class data.

**Wrong hypothesis 2 — seed non-determinism.** Verified that `ds.shuffle(seed=42)` in non-streaming mode works — produces 10 unique classes in first 10 samples. But downloading the full dataset consumed 50 GB (all three splits: train + val + test), filling Colab disk.

**Root cause.** The dataset has exactly 50 images per class in strict sequential order across all 52 shards. Any sampling strategy that takes the first N images gets the first N/50 classes only. Streaming shuffle is structurally ineffective for this dataset layout.

**Fix.** Strided sampling: `indices = list(range(0, step * max_samples, step))` where `step = 50000 // 2016 = 24`. This gives 2 images per class, deterministically, regardless of shuffle or session state. Verified: 46 unique classes in first 3 batches (96 images), consistent across runs.

---

## Known Limitations

### EfficientNet-B0 severely degraded under PTQ

**What fails:** 38% accuracy drop under fbgemm default observer; 22% under MinMaxObserver. Neither is acceptable for deployment.

**Root cause:** EfficientNet-B0's squeeze-and-excitation blocks use Sigmoid and Mul operators that have no INT8 equivalent in ONNX opset 17. These layers remain in FP32 in the exported graph (65 Sigmoid × 65 Mul residual nodes). The FP32/INT8 boundary at these operators introduces requantization error that compounds through the compound-scaled architecture.

**Concrete fix:** Model-specific observer configuration using `get_default_qconfig_mapping("fbgemm")` with per-layer overrides for SE blocks. Alternatively, QAT with SE blocks frozen in FP32.

### Calibration uses CIFAR-10, not ImageNet

**What fails:** Absolute accuracy numbers on CIFAR-10 were 0.00% for all models in early runs (ImageNet models evaluated against CIFAR-10 labels). Fixed by switching to ImageNet evaluation; calibration still uses CIFAR-10.

**Root cause:** The observer comparison experiment is designed to isolate calibration data quality as the single free variable. Using the same dataset for calibration and evaluation would confound the experiment. CIFAR-10 is used for calibration because it auto-downloads; ImageNet is used for evaluation because it has correct labels.

**Concrete fix:** Use an ImageNet subset for calibration as well. The infrastructure supports this — swap the `calibration_loader` argument in `run_pipeline()`.

### INT8 speedup requires explicit SIMD

**What fails:** C++ benchmark shows INT8 5–8× slower than FP32 on Apple Silicon. On x86, INT8 wins only at 128×128 (speedup 1.19) and loses at larger sizes.

**Root cause:** The FP32 inner loop auto-vectorises to NEON FMLA / AVX2 FMA. The INT8 inner loop has `static_cast<int32_t>` before each multiply and a 32-bit accumulator, breaking the auto-vectoriser's pattern recognition.

**Concrete fix:** Replace the inner loop with explicit NEON SDOT intrinsics on ARM or AVX2 VNNI on x86. Jacob et al. (2018) used gemmlowp — a hand-optimised integer kernel — not auto-vectorised C++. The absence of explicit SIMD is the finding; adding it would confirm H4 and separate the arithmetic question from the compiler question.

### H3 (miscalibration experiment) not run

**What fails:** The controlled experiment comparing real vs Gaussian calibration (`experiments/miscalibration.py`) was designed but not run end-to-end. The observer comparison experiment partially subsumes it.

**Concrete fix:** Run `experiments/miscalibration.py` with the fixed-seed calibration loader and ImageNet evaluation.

---

## What I Learned

**MinMaxObserver is not a safe default for production PTQ.** It is deterministic and requires no hyperparameters, which makes it an attractive baseline. But for architectures with wide or skewed activation distributions — particularly depthwise-separable and SE-gated networks — a single outlier in 100 calibration batches can destroy the quantization grid. `HistogramObserver` should be the default; MinMaxObserver is a controlled-experiment tool.

**A fixed quantization parameter is a latent bug that surfaces only at a different scale.** `S_C = 0.01` was plausible for a toy example and silently wrong for every real matrix size. Any scale factor set at design time rather than computed from data will fail when the data distribution changes. The lesson generalises: calibrated constants must be derived from representative data.

**The condition under which an empirical result holds is as important as the result itself.** H4 was stated on the basis of Jacob et al. (2018), which reports 2–3× INT8 speedups on ARM. The result is correct — under the condition that you use a hand-optimised integer kernel library (gemmlowp), not auto-vectorised C++. That condition was absent from the hypothesis. Stating it explicitly would have sharpened the experiment design.

**Dataset structure is a source of measurement error that is easy to miss.** The HF ImageNet dataset is sorted by class, 50 images per class, across 52 shards. Streaming shuffle with a buffer of 5000 covers only ~4 classes. This structural fact was not documented in the dataset card and required empirical discovery. Before trusting any evaluation dataset, verify the label distribution.

**PyTorch ONNX export breaks silently on version changes.** The default exporter changed from legacy to dynamo between minor PyTorch versions. The error was a crash, not a warning. `dynamo=False` is a required forward-compatibility guard that should be in every quantized model export call until the dynamo exporter explicitly supports quantized models.

**Observer sensitivity is architecture-dependent and must be measured, not assumed.** EfficientNet-B0 behaved worse under HistogramObserver than MinMaxObserver in the main benchmark run — an inversion of the expected result. The observer comparison experiment exists precisely to surface this kind of unexpected interaction. Assuming one observer is universally better is a design error.

---

## Results

| Dimension | ResNet-18 | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| FP32 size (MB) | 46.80 | 13.60 | 21.20 |
| INT8 size (MB) | 11.26 | 3.67 | 5.63 |
| Size ratio | 4.16× | 3.70× | 3.76× |
| FP32 latency — CUDA (ms) | 3.40 | 5.59 | 9.16 |
| INT8 latency — CPU (ms) | 36.76 | 21.74 | 46.89 |
| FP32 top-1 (ImageNet, strided) | 67.81% | 67.31% | 74.45% |
| INT8 top-1 (fbgemm) | 67.56% | 60.12% | 36.46% |
| Accuracy delta (fbgemm) | −0.25% | −7.19% | −38.00% |
| INT8 top-1 (minmax) | 64.93% | 0.10% | 52.63% |
| Accuracy delta (minmax) | −2.88% | −67.21% | −21.83% |
| Fits 4 MB budget | ✗ | ✓ | ✗ |
| ONNX coverage | 62.7% | 67.7% | 64.8% |

### Observer comparison finding

| Model | Δ fbgemm | Δ minmax | Observer sensitivity |
|---|---|---|---|
| ResNet-18 | 0.25% | 2.88% | Low — robust to both |
| MobileNetV2 | 7.19% | 67.21% | Critical — collapse under minmax |
| EfficientNet-B0 | 38.00% | 21.83% | High — degraded under both; worse under fbgemm |

### C++ scaling benchmark (H4 refuted)

| Size | Apple Silicon speedup | x86 speedup |
|---|---|---|
| 128×128 | 0.199 (INT8 slower) | 1.192 (INT8 faster) |
| 512×512 | 0.131 (INT8 slower) | 0.502 (INT8 slower) |
| 1024×1024 | 0.131 (INT8 slower) | 0.430 (INT8 slower) |

### Hypothesis outcomes

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | MobileNetV2 depthwise layers rank highest in L2 error | Partially consistent — prominent in top-10; EfficientNet shows higher absolute error |
| H2 | ~4× size reduction, 1.5–2.5× latency improvement | ✓ Confirmed — size 3.7–4.2×; latency comparison complicated by CPU/CUDA split |
| H3 | Real calibration outperforms Gaussian by ≥ 0.5% | Pending — miscalibration.py not run end-to-end |
| H4 | INT8 speedup grows with matrix size | Refuted on both platforms |

---

## What I'd Do Differently

**Verify the evaluation dataset's label distribution before running any accuracy measurement.** Three runs produced wrong results because the dataset's class ordering was assumed to be random. A five-line check — `Counter(ds["label"])`, `min/max counts`, `first 100 labels` — would have caught this in Phase 1 and saved multiple hours of debugging across Phases 2 and 3.

**Run the observer comparison experiment before choosing the default observer.** MinMaxObserver was chosen as the baseline because it is standard in the literature. The collapse it caused on MobileNetV2 was discovered only after running the full pipeline. Running a quick observer comparison on one model in Phase 2 would have identified the issue immediately and informed the design of the miscalibration experiment.

**Write the C++ benchmark with a small-scale sanity check before scaling.** Printing one element of `C_i8` and asserting `max(|C_i8|) < 127` before the full benchmark run would have caught the S_C clipping bug at the first call rather than after a hanging run and two wrong hypotheses.

**Use ImageNet for calibration from the start.** CIFAR-10 was chosen for its auto-download convenience. The correct calibration data — images from the same distribution the model was trained on — was available via HF streaming from Phase 2 onward. Using CIFAR-10 for calibration introduced an uncontrolled variable that required the observer comparison experiment to disambiguate.

---

## Status

All four phases complete. Observer comparison experiment complete. H3 (miscalibration experiment) pending. Calibration loader now has a fixed seed for reproducibility.

**Next three priorities if resumed:**
1. Run `experiments/miscalibration.py` end-to-end with fixed-seed calibration and ImageNet evaluation to populate H3
2. Add explicit NEON SDOT intrinsics to `bench.cpp` to confirm the theoretical INT8 speedup and separate the arithmetic question from the compiler question
3. Use ImageNet subset for calibration and measure whether observer collapse on MobileNetV2 persists with domain-matched calibration data

---

## References

- He et al., "Deep residual learning for image recognition," CVPR 2016
- Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," CVPR 2018
- Tan & Le, "EfficientNet: Rethinking model scaling for CNNs," ICML 2019
- Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," CVPR 2018
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference," arXiv 1806.08342, 2018
- Nagel et al., "A white paper on neural network quantization," arXiv 2106.08295, 2021

---

*See [README.md](../README.md) for setup instructions, project structure, and how to run.*
