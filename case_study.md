# EdgeZoo — Case Study

**Engineering decisions, failure modes, and what the project actually found.**

---

## What Makes This Hard

Post-training quantization is not a function call. The pipeline has five
sequential stages, each of which is a precondition for the next. Getting one
wrong does not produce an error — it produces a silently incorrect model.
The three specific hard problems in this project were:

**Calibration data quality is invisible until you instrument it.** A badly
calibrated model produces INT8 outputs that look plausible — the accuracy
drop is the only signal, and it is easy to attribute to quantization in general
rather than to calibration specifically. The miscalibration experiment was
designed to make this invisible variable visible by isolating it as the
single free variable.

**Quantization error does not distribute uniformly across layers.** A global
accuracy delta tells you that PTQ costs something — it does not tell you
where. Without layer-wise attribution, the degradation is a number with no
architectural explanation. The forward hook infrastructure was built
specifically to produce that explanation.

**The INT8 arithmetic advantage is real at the hardware level but conditional
on the implementation.** A compiler flag is not enough. The assumption that
`-O2 -march=native` would surface the SIMD throughput advantage of INT8 was
wrong, and the benchmark was designed to make that condition visible — not
to demonstrate the advantage, but to find where it does and does not
materialise.

---

## When to Use This (vs Alternatives)

✅ **Use this when:**
- You have a trained model, no access to the training pipeline, and a hard
  memory budget. PTQ is the correct tool when retraining is not an option.
- You need to understand *where* quantization costs before committing to a
  model choice, not just whether it works.
- You are preparing to hand a model to a hardware compiler team and need to
  document which operators will and will not be quantized.

❌ **Don't use this when:**

| Scenario | Why | What to use instead |
|---|---|---|
| Accuracy loss > 1% is unacceptable | PTQ has no recovery mechanism | QAT with full training pipeline |
| Target is a Transformer or LLM | Per-channel activation quantization and attention patterns require specialised treatment | GPTQ, AWQ, or SmoothQuant |
| You need real NPU latency numbers | All timing here is CPU-only | On-device profiling with NXP toolchain |
| Calibration dataset is unavailable | MinMaxObserver on random data produces degraded models | Synthetic calibration with domain statistics, or QAT |

---

## Industry Context and Positioning

| Alternative | What it does | Gap vs EdgeZoo |
|---|---|---|
| PyTorch quantization tutorial | Demonstrates PTQ on a single model | No failure analysis, no controlled experiment, no architectural comparison |
| Hugging Face Optimum | Production INT8 export for Transformers | CNN-focused, no layer-wise diagnostic, no C++ arithmetic benchmark |
| ONNX Runtime quantization | Graph-level quantization via onnxruntime tools | No calibration quality experiment, no operator coverage analysis |
| TensorFlow Lite converter | End-to-end mobile deployment pipeline | TF ecosystem only, no PyTorch FX graph mode, no diagnostic instrumentation |
| gemmlowp / XNNPACK | Hand-optimised integer kernel libraries | Library, not a diagnostic pipeline — does not explain where PTQ fails |

EdgeZoo is not a deployment tool. It is a diagnostic pipeline. The distinction
matters: a deployment tool optimises for a metric; a diagnostic pipeline
optimises for understanding the tradeoffs that determine which metric is
achievable. The project produces findings, not a production artifact.

---

## Technical Approach

### The data contract

Every downstream function receives a `ModelEntry` from the registry rather
than loading a model directly. This means the pipeline, the benchmarks, and
the error attribution all operate on a common interface regardless of
architecture. The registry stores `arch_family` as an explicit field because
the architectural label is the explanatory variable for H1 — without it, the
attribution results are numbers without a frame.

### The PTQ pipeline as a sequencing constraint

The five stages are not a design choice — they are a constraint. BN-folding
must precede observer insertion to prevent double-scaling of normalisation
parameters. Calibration must precede INT8 conversion because conversion reads
the observer ranges. ONNX export must follow conversion because the INT8
operators only exist in the converted graph. Writing the pipeline as
composable functions makes the sequencing explicit and auditable.

### The observer capture window

Observer ranges — the `[α, β]` per-layer min/max values — exist only between
Stage 3 (calibration) and Stage 4 (conversion). After `convert_fx()`, observer
nodes are replaced with static scale and zero-point constants. The miscalibration
experiment captures ranges in this window specifically. Missing it means the
comparison is impossible to reconstruct without re-running the full pipeline.

---

## Key Engineering Decisions

### PTQ over QAT

**Chosen:** Post-training quantization with no retraining.
**Rejected:** Quantization-aware training.
**Why:** No access to the training pipeline or labelled training data; one-day
constraint. QAT achieves higher final accuracy by exposing the model to
quantization noise during training, but it requires training epochs, a full
dataset, and a loss function. PTQ operates on a frozen model. Under the stated
constraints, PTQ is not a simplification — it is the only viable option.
Framing it as a simplification would be incorrect and would prevent a
productive discussion about when QAT is worth its cost.

### MinMaxObserver as the baseline observer

**Chosen:** `MinMaxObserver` — records the minimum and maximum activation value seen during calibration.
**Rejected:** `HistogramObserver` (percentile-based), KL-divergence observer.
**Why:** MinMaxObserver is deterministic, has no hyperparameters, and its
sensitivity to outliers is a feature of the miscalibration experiment rather
than a defect. An observer with built-in outlier suppression would partially
compensate for bad calibration data and reduce the signal the experiment is
designed to isolate. Percentile and KL observers are a natural next step and
are explicitly deferred.

### Per-channel weights, per-tensor activations

**Chosen:** Per-channel quantization for weights; per-tensor for activations.
**Rejected:** Per-tensor for both; per-channel for both.
**Why:** Weight distributions vary substantially across output channels —
a single scale per layer would waste INT8 resolution on channels with narrow
ranges to accommodate channels with wide ones. Per-channel gives each output
channel its own scale. Activation distributions are more uniform within a
layer; per-tensor avoids the runtime cost of per-channel scale vectors at
inference. This is the configuration recommended in Krishnamoorthi (2018)
and is standard for production PTQ pipelines.

### Dynamic S_C in the C++ benchmark

**Chosen:** Output scale computed from `max(|C_fp32|) / 127` per run.
**Rejected:** Fixed `S_C = 0.01`.
**Why:** The output magnitude of a matrix multiply scales with K. At K = 128,
values reach ±300; at K = 1024, ±2400. A fixed S_C of 0.01 maps the output
grid to [-1.28, 1.27] — nearly every output element is clipped to ±127 and
the error measurement is meaningless (it reports the clipping error, not
the quantization error). Computing S_C from the actual FP32 output makes
the grid correct at every size independently. This was a bug discovered
during the benchmark run and is documented below.

### ONNX interrogation over verification

**Chosen:** Programmatic node classification — quantized vs FP32 arithmetic vs structural.
**Rejected:** Passive verification (checking scale parameter correctness only).
**Why:** Verification confirms that the quantization was applied correctly to
the operators it was applied to. Interrogation asks which operators it was
*not* applied to and why. The second question is the one a hardware compiler
engineer would ask: which subgraphs would the toolchain reject, and what
would you do about them? Each FP32 residual node in the ONNX graph is a
finding, not a formatting detail.

### Standard C++ only in the benchmark

**Chosen:** No external libraries — only `<chrono>` and `<cstdint>`.
**Rejected:** OpenBLAS, BLAS Accelerate, explicit NEON intrinsics.
**Why:** An optimised library would produce the expected INT8 speedup but
would also obscure the condition under which the advantage materialises.
The benchmark is designed to measure what a compiler produces from a readable
implementation — not what a hand-optimised kernel achieves. That distinction
*is* the finding. Using NEON SDOT or AVX2 VNNI explicitly would confirm H4
rather than test it.

---

## The S_C Bug — A Forensic Account

**Phase:** 3 — C++ benchmark, first run on Apple Silicon.

**Symptom:**

```
Size    FP32 (ms)  INT8 (ms)  Speedup  Max Abs Error
128×128  0.155      0.717      0.217    93.466
512×512  12.653     107.712    0.117    215.563
[benchmark hanging — never reaches 1024×1024]
```

Two problems: the error of 93.466 is not in a plausible range for quantization
error, and the benchmark hung after the second row.

**Wrong hypothesis 1 — formatting issue.** The error looked like a unit
mismatch or a scale factor applied twice. Inspected `max_abs_error()` — the
function was correct. The dequantization was `C_i8[i] * S_C`, which is right.
Ruled out.

**Wrong hypothesis 2 — the INT8 output was all zeros.** If the quantized
matmul was producing zeros, the error would equal `max(|C_fp32|)` — plausible
given the magnitude. Printed a slice of `C_i8`. The outputs were ±127 almost
everywhere, not zero. The opposite problem: saturation, not silence.

**Root cause.** `S_C` was fixed at `0.01`. The output grid therefore covered
`[-1.28, 1.27]`. The actual output of a 128×128 matmul with inputs in
`[-2.54, 2.54]` reaches approximately `±300`. The requantization step
multiplied every INT32 accumulator by `S_A * S_B / S_C = 0.04`, producing
scaled values in the thousands — all clipped to `±127`. The error of 93
was not a quantization error; it was a measurement of how badly the output
was clipped. The fix: compute `S_C = max(|C_fp32|) / 127` from the actual
reference output before each run.

**The hang.** Separately, the benchmark used 1000 repetitions at all three
sizes. At 1024×1024 with a naive triple-loop, one forward pass takes ~96 ms
on Apple Silicon. 1000 repetitions would take ~96 seconds at that size alone.
The benchmark was not hung — it was simply running. Fixed by scaling reps:
1000 / 100 / 10.

**What would have found it faster.** Printing one element of `C_i8` before
running the full benchmark would have shown saturation immediately. A sanity
check — `assert max(|C_i8|) < 127` — would have caught it at the first run.
For any benchmark that measures error, the error should be inspected on a
small case before scaling up.

**Architectural implication.** Any scale factor that depends on the magnitude
of a matmul output must be computed dynamically per problem size. A fixed scale
is a latent bug that becomes visible only when the problem size changes. In
production, this is why deployment recipes store scale factors per-layer as
calibrated constants, not as compile-time parameters.

---

## Known Limitations

### Accuracy evaluation on CIFAR-10, not ImageNet

**What fails:** The absolute top-1 accuracy numbers reported by the pipeline
are lower than published ImageNet benchmarks because the models were evaluated
on CIFAR-10 resized to 224×224, not ImageNet. The models were not trained on
CIFAR-10 and their feature detectors are not calibrated for it.

**Root cause:** ImageNet requires a manual download and a licence agreement.
CIFAR-10 downloads automatically. The choice was made to prioritise
reproducibility over domain-exact evaluation.

**Concrete fix:** Replace `observers.py`'s `real_calibration_loader()` with
an ImageNet validation subset loader. The accuracy delta (FP32 vs INT8 on
the same set) remains valid regardless of which dataset is used — only the
absolute numbers change.

### MinMaxObserver sensitivity to outliers

**What fails:** A single extreme activation value encountered during
calibration stretches the INT8 grid to accommodate it, wasting resolution
for the bulk of the distribution. This is particularly acute for small
calibration sets.

**Root cause:** MinMaxObserver records `min` and `max` with no outlier
suppression. The correct fix is a percentile observer (e.g. 99.9th percentile)
or a KL-divergence observer that minimises the information loss of the
quantization.

**Concrete fix:** Replace `MinMaxObserver` with
`torch.ao.quantization.observer.HistogramObserver` in `pipeline.py`'s
`qconfig` definition. The miscalibration experiment infrastructure already
supports this — it is a one-line change.

### No real NPU latency

**What fails:** All latency measurements are CPU wall-clock time. The C++
benchmark demonstrates the arithmetic efficiency argument but cannot report
actual inference latency on an NPU.

**Root cause:** Hardware not available. NXP Ara-2 is not publicly accessible.

**Concrete fix:** Export the quantized ONNX model and run it through the
eIQ Toolkit with NXP's profiling tools. The ONNX graph interrogation
infrastructure already identifies which subgraphs the compiler would accept.

### INT8 speedup requires explicit SIMD, not just `-march=native`

**What fails:** The C++ benchmark shows INT8 slower than FP32 on Apple Silicon
at all sizes, and slower at large sizes on x86. The theoretical 4× throughput
advantage of INT8 SIMD does not materialise from compiler-generated code.

**Root cause:** The FP32 inner loop (`float += float * float`) is a pattern
the compiler recognises and maps to FMLA instructions. The INT8 inner loop
has `static_cast<int32_t>` before each multiply and a 32-bit accumulator —
enough structural complexity that the auto-vectoriser does not produce an
equivalent SIMD reduction.

**Concrete fix:** Replace the inner loop with explicit NEON SDOT intrinsics
on ARM or AVX2 VNNI on x86. This would confirm the theoretical speedup and
separate the arithmetic question from the compiler question.

---

## What I Learned

**A fixed quantization parameter is a latent bug that only surfaces at a
different scale.** `S_C = 0.01` was correct for a toy example; it was wrong
for every real matrix size in the benchmark. Any scale factor that is set
at design time rather than computed from data will fail when the data changes.
The lesson for system design: calibrated constants must be derived from
representative data, not chosen once and hardcoded.

**The condition under which an empirical result holds is as important as the
result itself.** H4 was stated on the basis of Jacob et al. (2018), which
reports 2–3× INT8 speedups on ARM. The result is correct — under the condition
that you use a hand-optimised integer kernel library. That condition was not
stated in H4, and the experiment made it visible. Before citing a paper's
empirical result in a hypothesis, identify what implementation conditions
the result assumes.

**A refuted hypothesis is a better interview answer than a confirmed one.**
H4 being wrong produced a finding: the INT8 arithmetic advantage requires
explicit SIMD or a dedicated integer pipeline, not just a compiler flag. That
finding directly motivates why NPUs exist. A confirmed H4 would have been a
number. A refuted H4 is an argument.

**Isolating one variable requires holding all others constant, and that
requires infrastructure.** The miscalibration experiment is only valid because
both calibration conditions use the same architecture, observer type, bit-width,
and evaluation data. Building that infrastructure — the shared fixed-seed
evaluation loader, the observer range capture window — takes as much time as
the experiment itself. The experiment design is part of the engineering work.

**Squiggles in VS Code are not errors.** `std::clamp` is valid C++17.
IntelliSense not recognising it is a configuration problem, not a code
problem. Build and run first; fix the editor configuration separately.
Stopping to debug the IDE before compiling inverts the priority.

---

## Results

| Dimension | FP32 baseline | INT8 PTQ |
|---|---|---|
| ResNet-18 size | 46.8 MB | 11.7 MB (4.0× reduction) |
| MobileNetV2 size | 13.6 MB | 3.4 MB (4.0× reduction) |
| EfficientNet-B0 size | 21.2 MB | 5.3 MB (4.0× reduction) |
| Fits 4 MB budget | None | MobileNetV2 only |
| INT8 latency vs FP32 (CPU) | — | [fill after Phase 2 run] |
| Accuracy delta (real calibration) | — | [fill after Phase 2 run] |
| Calibration delta (real vs Gaussian) | — | [fill after Phase 2 run] |
| INT8 speedup — Apple Silicon 128×128 | 1.0× | 0.20× (FP32 faster) |
| INT8 speedup — x86 128×128 | 1.0× | 1.19× (INT8 faster) |
| INT8 speedup — x86 1024×1024 | 1.0× | 0.43× (FP32 faster) |

---

## What I'd Do Differently

**Write the calibration sanity check before the full benchmark run.** A
two-line assertion — print one element of the INT8 output, check that
`max(|C_i8|) < 127` — would have caught the S_C bug at the first run
rather than after a hung benchmark and a wrong-hypothesis cycle. For any
experiment that measures error, add a small-scale sanity check before
scaling up.

**State the implementation conditions in each hypothesis.** H4 said "INT8
speedup grows with matrix size." It should have said "INT8 speedup with
compiler-generated code grows with matrix size, assuming the compiler
auto-vectorises the INT8 inner loop." The missing condition is the entire
finding. Stating it explicitly in the hypothesis would have sharpened the
experiment design.

**Run Phase 2 in Colab before writing the Phase 2 report.** The accuracy
delta and layer error rankings are still null in the recipe because Phase 2
was designed and documented before being run end-to-end. In a real project,
results tables are written after the experiment, not before. The structure
was right; the sequencing was wrong.

---

## Status

Phase 3 complete. Phase 4 (deployment recipe and README) complete.
Phase 2 results (`benchmark/bench.py`) pending full Colab run to populate
accuracy delta, layer error rankings, and ONNX coverage table.

**Next three priorities if resumed:**
1. Run `benchmark/bench.py` in Colab and fill null fields in `recipe.yaml` and this document
2. Replace `MinMaxObserver` with `HistogramObserver` and re-run the miscalibration experiment to measure the delta
3. Add explicit NEON SDOT intrinsics to `bench.cpp` and compare against the auto-vectorised baseline

---

## References

- He et al., "Deep residual learning for image recognition," CVPR 2016
- Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," CVPR 2018
- Tan & Le, "EfficientNet: Rethinking model scaling for CNNs," ICML 2019
- Jacob et al., "Quantization and training of neural networks for efficient
  integer-arithmetic-only inference," CVPR 2018
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient
  inference," arXiv 1806.08342, 2018
- Nagel et al., "A white paper on neural network quantization,"
  arXiv 2106.08295, 2021

---

*See [README.md](../README.md) for setup instructions, project structure, and
how to run.*