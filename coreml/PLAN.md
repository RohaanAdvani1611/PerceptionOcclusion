# MiDaS on Core ML: implementation plan

**Status: approved on 2026-11-01. Implementation starts on your go-ahead.**

**Goal:** convert MiDaS v2.1 small to Core ML, compare four compressed variants on size, latency, accuracy and compute-unit placement, and run them on live camera frames in a small Swift app, on the M2 Mac first and then the iPhone 14.

All new work stays in `coreml/`. The only existing file this plan changes is `.gitignore`.

## Facts this plan is built on

| Item | Value | How it affects the plan |
|---|---|---|
| Mac | MacBook Air M2 (`Mac14,15`), 8 GB, macOS 15.6.1 | It has no fan, so long runs will likely slow down from heat. Memory is tight, so close other apps while benchmarking. |
| iPhone | iPhone 14 (A15), iOS 27.0.1 | Meets the iOS 17.4 minimum for per-op placement data. It's newer than Xcode 26.3's iOS 26.2 SDK, but a test app installed and ran on it, so no upgrade is needed. |
| Xcode | 26.3 with Swift 6.2.4; first-launch setup is done | No install needed. |
| Default Python | 3.15.1 | Too new for coremltools, so we use a separate 3.12 environment. |
| coremltools | 9.0, the latest; wheels up to Python 3.13; notes list PyTorch 2.7 | We pin PyTorch to 2.7, not the latest 2.15. |
| MiDaS weights | `midas_v21_small_256.pt`, 85.8 MB, available upstream | The source of the conversion. |
| Current speed | `session.Run` about 40 ms in a separate probe; live `infer` 44.8 ms mean, 57.5 ms p99 | The numbers to beat. Compare model time with model time. |

## Apple concepts used below

- **`.mlpackage`:** the folder Core ML uses to store a model, holding its operations and weights. It's what we produce from Python and open in Xcode.
- **ML Program:** the modern Core ML model format, required for the compression features used here.
- **Compute units:** a setting that tells Core ML which processors it may use. `.all` lets it choose, and the others restrict it. It's permission, not a command, so an op can still fall back to CPU.
- **Apple Neural Engine (ANE):** the dedicated ML processor on your M2 and A15. It's usually efficient but only runs certain ops and shapes.
- **`CVPixelBuffer`:** Apple's image container. When backed by an IOSurface (memory shared between processors), it can move between the camera, ANE and GPU without copying.
- **`MLComputePlan`:** an API that reports, for each op, the processor Core ML prefers, the ones it supports, and an estimated cost. Available on macOS 14.4 and iOS 17.4 or later.

## How the work is split

Each phase ends with a check for you to confirm before the next starts.

| Phase | Work | Estimate |
|---|---|---|
| 3 | Environment, MiDaS source, capture frames, conversion, numerical check | 2–4 hours |
| 4 | ANE placement analysis | 1–3 hours |
| 5 | Compress, evaluate accuracy | 2–3 hours |
| 6 | Swift app on Mac, with live view and benchmark mode | 3–5 hours |
| 7 | ONNX timing runner, Mac benchmarks, performance reports, soak test | 2–3 hours |
| 8 | Deploy to iPhone 14 and compare | 2–4 hours |
| 9 | `RESULTS.md` and talking points | 1–2 hours |
| Optional | Temporal filter or occlusion demo in the app | 2–4 hours |

Most of the uncertainty is in Phase 4 and in signing during Phase 8.

## 1. Conversion

**Setup.** Create `coreml/.venv` with Python 3.12 and pinned versions: `torch==2.7.0`, `coremltools==9.0`, and the other packages listed in `coreml/requirements.txt`. Exact source versions are in `coreml/VERSIONS.md`.

**Source.** Clone the official MiDaS repo into `coreml/third_party/`, record the commit, and build `MidasNet_small` directly with the official weights. Its backbone comes from `rwightman/gen-efficientnet-pytorch` through `torch.hub`, so that revision gets pinned too.

**Path.** Use `torch.jit.trace`, then `ct.convert(..., convert_to="mlprogram")`. Tracing is the most established path, and the coremltools notes describe `torch.export` as still catching up on coverage. Nothing about this model needs `torch.export`.

**Fixed settings.**
- Input shape 1×3×256×256, matching the current pipeline. Fixed shapes also help the ANE.
- `minimum_deployment_target=iOS17`, because W8A8 needs it. The same target applies on both devices.
- `compute_precision=FLOAT16` for the baseline, plus a FLOAT32 reference used only for checks.
- `model.eval()` before tracing, so BatchNorm uses its stored statistics.

**Normalization must happen exactly once.** Phase 0 found this bug in the old pipeline. The PyTorch model has no built-in normalization, so a small wrapper adds it.

**Input type decision: image, not tensor.**

| | Image input | Tensor input |
|---|---|---|
| App passes | `CVPixelBuffer` | `MLMultiArray` |
| Pixel handling | Core ML converts the format and applies scale and bias | We convert and normalize by hand on the CPU |
| Copies | Can avoid them with IOSurface-backed buffers | Usually at least one |
| Flexibility | Limited | Full |

We use image input because it suits a camera app and is what you came to learn. There are two limits to know about:

1. **Core ML won't resize for you.** Its docs only describe Vision doing automatic resizing. With direct Core ML, the buffer must already be 256×256.
2. **`ImageType` has one global `scale`, but ImageNet has three std values.** So `ImageType` applies only `scale=1/255`, and the wrapper applies mean and std as ops inside the model, which is what the ONNX export did.

**Output type decision: `GRAYSCALE_FLOAT16` image.** This gives a `CVPixelBuffer` that Metal can draw directly, and needs a 1×1×256×256 output with no final squeeze. If it causes trouble, the fallback is an `MLMultiArray`.

**Numerical check (`convert.py`).** One preprocessing path feeds saved 256×256 frames to all models, so differences come from the models and not from resizing.

| Comparison | Expected | Purpose |
|---|---|---|
| PyTorch vs Core ML FP32 on CPU | Pearson above 0.9999 | Proves the conversion is right |
| FP32 vs FP16 on CPU | Pearson above 0.999 | Measures the precision loss |
| FP16 on CPU vs GPU vs ANE | Report the values | Different processors give slightly different numbers |
| PyTorch vs corrected ONNX | Report the values | Measures the resize gap. An early check on 8 demo frames found a Pearson of about 0.956, even though the shared weights are identical |

These thresholds are starting targets I chose, not published limits. If the first comparison fails, nothing else proceeds.

**Results (2026-11-02, `check_conversion.py`, all 250 captured frames):**

| Comparison | Pearson mean | Worst frame | Status |
|---|---|---|---|
| PyTorch vs Core ML FP32 on CPU | 1.000000 | 1.000000 | Pass |
| FP32 vs FP16 on CPU | 0.999861 | 0.997267 | Below target on 7 frames |
| FP32 vs FP16 on CPU and GPU | 0.999976 | 0.999211 | Pass |
| FP32 vs FP16 on CPU and Neural Engine | 0.999976 | 0.999169 | Pass |
| PyTorch vs the C++ pipeline's ONNX | 0.974415 | 0.866563 | The resize gap, worst on a blank wall |

- **FP16 on the CPU alone** loses more precision than on the GPU or Neural Engine, as small errors spread across the whole image, with no overflow. Task 14 investigates this. The baseline stays FP16, since the processors the app uses pass.
- **Tracing:** the backbone caches its "same" padding sizes on its first run, so `convert.py` runs the model once before tracing. Otherwise the first trace records shape arithmetic and PyTorch's trace check fails.
- **Size:** the FP16 package is 33.3 MB, confirming that the unused 4.7M-parameter block is dropped.

**Phase 3 check:** the `.mlpackage` opens in Xcode showing the expected inputs and outputs, the script prints passing values, and a side-by-side depth image looks right.

## 2. Neural Engine fit

**How we find out where ops run.** We use three sources, because no single one gives the full picture:

1. **`MLComputePlan`** lists each op's preferred and supported devices and its estimated cost, which produces the "% on ANE" figure. Phase 4 uses the coremltools Python bindings, because the app doesn't exist yet. If they don't give enough detail, a small standalone Swift tool will.
2. **Xcode's performance report** gives the same picture visually, with measured load and prediction time.
3. **Measured latency under each compute unit.** If `.cpuAndNeuralEngine` is about as fast as `.cpuOnly`, the ANE isn't doing much, whatever the plan says.

**Report "% on ANE" two ways:** by op count and by estimated cost. A model can be 95% ANE by count and still spend most of its time on a few CPU ops. The plan is also a prediction, so we check it against measured latency.

**Where each op type stands.** These are things to test, not predictions. The model is Conv, BatchNorm, ReLU6, ReLU, Add, Pad and bilinear Resize at a small fixed size, which is generally favourable.

| Possible cause of fallback | Applies here? | What we'd try if it falls back |
|---|---|---|
| Asymmetric `Pad` before stride-2 convs | Yes, 5 in the ONNX graph | Replace with equivalent padding, then recheck accuracy |
| Bilinear with `align_corners=True` | Yes, 4 fusion blocks | Set to `False`, then measure the accuracy cost |
| Dynamic shapes | No, all fixed | Nothing |
| Very large tensors | Unlikely at 256×256 | Nothing |
| Unsupported ops | To be measured | Rewrite them in the wrapper |
| Costly CPU-to-ANE switching | To be measured | Move or rewrite the op that causes the split |

Any model change here becomes its own named experiment, with accuracy measured, so the baseline stays clean. `.all` may also choose the GPU over the ANE if Core ML estimates it's cheaper, which is why we test `.cpuAndNeuralEngine` separately.

**FP16 check.** In the Phase 0 run with correct normalization, raw outputs mostly peaked at about 1,000–1,500. FP16 handles that easily, with a step of 1.0 between 1,024 and 2,048, so the final output isn't the worry. An intermediate layer could still overflow or lose precision, so we check for:
- `inf` or `NaN` in any output
- The FP32 vs FP16 comparison on the worst frame, not only the average
- If needed, intermediate activations compared layer by layer

**Phase 4 check:** a per-op table showing the device for each op and an explanation for each one not on the ANE.

**Results (2026-11-02, FP16 on the M2):**

| Model | Compute units | Ops on ANE | Cost on ANE | Device switches |
|---|---|---|---|---|
| First baseline, `scale=1/255` on the image input | `.cpuAndNeuralEngine` and `.all` | 202 of 204 (99.0%) | 98.7% | 1 |
| **Current baseline**, 1/255 inside the model | `.cpuAndNeuralEngine` and `.all` | **202 of 202 (100%)** | **100%** | **0** |

`.all` chose the ANE over the GPU in both.

**Xcode performance report, current baseline, M2 on macOS 15.6.1, one run each (2026-11-02):**

| Compute units | Prediction (median) | Load (median) | Compilation (median) | Ops: CPU / GPU / ANE |
|---|---|---|---|---|
| All | 2.20 ms | 26.54 ms | 94.10 ms | 0 / 0 / 202 |
| CPU and Neural Engine | 1.80 ms | 23.03 ms | 95.85 ms | 0 / 0 / 202 |

Xcode agrees with `MLComputePlan`: every op is on the ANE. Against the ONNX baseline, `session.Run` took about 40 ms and loading took 97.5 ms. The 0.4 ms gap between the two settings comes from a single run each, so Stage 6 checks it with repeated measurements before drawing a conclusion.

**Baseline decision (2026-11-02):** the 1/255 scale moved into the model, making it 100% ANE. The first baseline is kept as `models/experiments/midas_small_fp16_scale_outside.mlpackage` for the before-and-after comparison. The Stage 2 numbers above are from the first baseline. With the current one, on the ANE against FP32 across all 250 frames: Pearson mean 0.999978 (was 0.999976), closer to FP32 on 152 of 250 frames, and mean absolute error 1.10 (was 1.21). Its worst frame, `bedroom_0042`, fell from 0.99930 to 0.99819, the only frame below 0.999 on the GPU and ANE. That frame is also the worst on the CPU, where the worst is now 0.99633 on 11 frames (was 0.99727 on 7). A simulation of the FP16 normalization step shows both versions err by about 0.0003 on average, so no single step got worse: the rounding falls differently, and the final layer magnifies it on a few frames.

- **Every model op is on the ANE,** including the 5 asymmetric `pad` ops and the `align_corners=True` resizes this plan worried about. Conversion folded all 72 BatchNorms into convolutions. Per-op table: `results/placement_midas_small_fp16.csv`.
- **The only 2 CPU ops are Core ML's own input handling.** `ImageType(scale=1/255)` makes Core ML scale the image in FP32 (`mul`) and then convert it to FP16 (`cast`), and the ANE runs neither. Folding the 1/255 into the model's normalization removes both, giving 202 of 202 ops on the ANE with unchanged accuracy (worst frame 0.99967 against the baseline). In Python it saved only 0.03 ms of about 1.96 ms, which is within noise. An FP16 tensor input also removes them and saved 0.10 ms, but moves frame conversion into the app (`experiment_input_ops.py`).
- **FP16 overflow:** none. The largest activation over all 250 frames is 7,277, in the output head, against FP16's limit of 65,504. The backbone stays under 124.
- **Why FP16 on the CPU is less accurate:** on the CPU, an FP16 convolution's error grows with the number of inputs it sums: 0.04% at 16, 0.7% at 1,024 and 7% at 4,096, matching a simulated FP16 running sum. On the GPU and ANE it stays at about 0.02%, the rounding of the result alone, so they sum in higher precision. Comparing each stage of the network shows the CPU's extra error from the first backbone stage onwards. Every device loses most at the final 32-to-1 convolution, where large positive and negative values cancel.
- **Small ops go to the CPU even when the ANE is allowed:** a lone 1×1 convolution on a 16×16 input stayed on the CPU under `.cpuAndNeuralEngine`, which is Core ML's cost model at work.
- **Wrapper arithmetic order matters in FP16:** subtracting the mean before multiplying gave a CPU worst frame of 0.9973, and multiplying first gave 0.9929. The wrapper subtracts first.

## 3. Compression variants

Everything is built from the same FP16 model using `coremltools.optimize.coreml`. Sizes are my estimates from the 16.6M parameters that `forward()` actually uses. The checkpoint has 21.3M, but one 4.7M block is never run. Real numbers replace them in `RESULTS.md`.

| Variant | How | Data needed | Estimated size |
|---|---|---|---|
| Baseline FP16 | `compute_precision=FLOAT16` | None | ~33 MB |
| 8-bit weights | `linear_quantize_weights` | None | ~17 MB |
| 6-bit palettized | `palettize_weights`, `nbits=6`, `mode="kmeans"` | None | ~13 MB |
| W8A8 | `linear_quantize_activations`, then `linear_quantize_weights` | 100–200 frames | ~17 MB |

**What to expect on your hardware.** Coremltools' docs say int8-int8 ANE compute arrives from iPhone 15 Pro onwards, so neither the M2 nor the A15 has it. W8A8 is an experiment. It may run with no speedup, or even more slowly, and it may lose accuracy. Any of those is a useful result if we can explain it.

**Calibration frames.** These are captured in Phase 3, because the conversion check needs them too. `coreml/capture_frames.py` saves frames from your webcam, from a few different scenes and lighting conditions.
- **Calibration set:** 150 frames. These set the activation ranges.
- **Held-out set:** 100 frames from different scenes. Used only for accuracy, since testing on calibration frames would flatter W8A8.
- **Privacy:** these are pictures of your room, so they stay local and are gitignored.

**API note.** The docs list activation quantization under `cto.coreml.experimental` and don't spell out the `sample_data` format. We confirm both against the installed 9.0 package.

## 4. Accuracy evaluation

**Reference:** PyTorch FP32 on the held-out frames. Not the ONNX model, since it differs in its resize steps.

**Metrics, each computed per frame:**

| Metric | What it answers |
|---|---|
| AbsRel after least-squares scale and shift alignment | How large is the error, ignoring MiDaS's unknown scale and shift? |
| Pearson correlation | Does the overall depth pattern match? |
| Occlusion mask disagreement at threshold 0.5 | Would the app's occlusion actually change? |

We report the mean and the worst frame. AbsRel divides by the reference depth, and in Phase 0 it blew up on near-zero values, so it excludes pixels below the reference's 1st percentile. The mask metric is the one specific to this project, and probably the most useful in an interview.

**Visual output:** `coreml/results/comparisons/` gets side-by-side images of input, reference, each variant and an error heatmap, including each variant's worst frame.

**Phase 5 check:** four `.mlpackage` files exist with real sizes, plus an accuracy table and images you can inspect.

**Results (2026-11-03, 100 held-out frames, `.cpuAndNeuralEngine`, against PyTorch FP32; `results/accuracy.csv`):**

| Version | Size | Broken frames | AbsRel mean (worst) | Mask flips mean (worst) | Python p50 |
|---|---|---|---|---|---|
| FP16 | 33.3 MB | 0 | 0.23% (1.7%) | 0.17% (3.1%) | ~1.9 ms |
| W8, head FP16 | 16.9 MB | 0 | 1.68% (10.3%) | 1.05% (11.9%) | ~1.9 ms |
| P6 | 12.7 MB | 0 | 7.43% (27.0%) | 4.58% (22.9%) | ~1.75 ms |
| W8A8 | 17.0 MB | 0 | 12.01% (42.4%) | 8.91% (86.8%) | ~1.55 ms |
| ONNX (C++ pipeline) | — | 0 | 12.45% (58.9%) | 6.73% (43.3%) | — |

All Core ML versions run 100% on the ANE. Python timings are rough, and Stage 6 measures properly.

- **W8 with every weight 8-bit breaks on the ANE:** 48 of 100 frames contain inf or NaN, though the same model is fine on the CPU and GPU. Probing each stage shows the failure starts in the output head's two 3×3 convs, which carry the network's largest activations. A test of whether the ANE sums whole-number weights before scaling predicted overflow on every frame, and was ruled out. Keeping those two convs FP16 fixes it for about 28 KB (`compress.py`). The naive version is kept as `experiments/midas_small_w8_naive`.
- **Losses come from the compression, not the ANE:** each version's accuracy is about the same on the ANE, GPU and CPU.
- **AbsRel exaggerates far regions:** it divides by inverse depth, which is small far away. W8's and P6's worst frames show errors in a far patch with almost no mask flips. Mask flips is the measure that matters for the app.
- **W8A8 loses its accuracy in its activations, not its weights**, and the cause is unresolved. A8 alone (FP16 weights) is as bad. Keeping FP16 activations in the output head, the whole decoder, or every `add` barely helps (8.1–9.1% flips), so the loss is in the backbone. A PyTorch simulation of coremltools' rule, which rounds only the inputs of `conv`, `add` and pooling ops, reaches only 2.2% flips against Core ML's 8.5%, so it misses the cause. Investigation stopped there (`experiment_w8a8.py`).
- **W8A8 was the fastest on the ANE** in every Python timing (about 1.55 ms against 1.9 ms), despite 387 extra quantize and dequantize ops, and the slowest on the GPU (8.5 ms against 7.3 ms). This needs confirming in Swift.
- **P6 with one palette per 16 output channels** halves P6's error (2.41% flips) but grows to 15.9 MB, nearly W8's size, while W8 is more accurate. It needs the iOS 18 / macOS 15 format, and 11 tensors couldn't be grouped and stayed FP16. Kept as an experiment (`experiment_p6.py`).
- **Setup:** scikit-learn 1.9.1 was added for k-means palettization. coremltools' warning that it supports only up to 1.5.1 concerns its scikit-learn model converter, which isn't used here.

## 5. Benchmarking

**Measure in Swift, not Python.** Python adds its own overhead, and an app is what would ship. One benchmark mode in the app runs on both devices, so the numbers are comparable.

For **each variant × each compute unit** (`.all`, `.cpuAndNeuralEngine`, `.cpuAndGPU`, `.cpuOnly`), on real held-out frames:

| Measure | How |
|---|---|
| File size | Compiled `.mlmodelc` and `.mlpackage` sizes |
| Load time | First load and cached load, since the first ANE load compiles the model and can be far slower |
| Latency | 30 warm-up runs skipped, then 500 timed, reporting p50, p90, p99 |
| Conditions | Plugged in, Low Power Mode off, thermal state logged, cooldown between runs |

That's 16 combinations per device, and CSVs go to `inference/coreml/`. Roughly 15–30 minutes per device.

**Mac results (2026-11-21, Release build, plugged in, Low Power Mode off, thermal state nominal throughout; `inference/coreml/benchmark_mac_m2_2026-11-21.csv`).** Prediction p50 in ms, 500 timed runs on the 100 held-out frames:

| Version | All | CPU + Neural Engine | CPU + GPU | CPU only | First load, CPU+NE | Second load |
|---|---|---|---|---|---|---|
| FP16 | 1.83 | 1.81 | 8.05 | 11.08 | 52 ms* | ~20 ms |
| W8 | 1.69 | 1.69 | 8.47 | 11.13 | 1,054 ms | ~23 ms |
| P6 | 1.59 | 1.59 | 8.43 | 11.10 | 740 ms | ~22 ms |
| W8A8 | 1.41 | 1.43 | 8.48 | 17.31 | 2,075 ms | ~34 ms |
| ONNX Runtime, CPU, 4 threads | | | | 39.69 | 107 ms | |

\*FP16's model was already compiled for the Neural Engine from earlier runs. Its first load under `.all` in this session took 775 ms.

- **On the ANE, smaller is faster:** FP16 1.81 ms, W8 1.69, P6 1.59, W8A8 1.43. W8A8, the smallest in flight because its activations are 8-bit too, is 21% faster than FP16 despite 387 extra ops. This fits the ANE being limited by moving data rather than by arithmetic, but it's an inference from timing, not a measurement of memory traffic.
- **On the GPU and CPU, compression doesn't help:** the compressed versions are the same speed or slightly slower, and W8A8 on the CPU is 56% slower (17.3 ms against 11.1), because the CPU runs every quantize and dequantize op.
- **`.all` matches `.cpuAndNeuralEngine`,** so the 0.4 ms gap in the single Xcode report was noise.
- **Against ONNX:** Core ML FP16 on the ANE is 22× faster than the C++ pipeline's model (1.81 ms against 39.69 ms), and W8A8 is 28× faster. Even on the CPU alone, Core ML FP16 is 3.6× faster than ONNX Runtime FP32 with 4 threads.
- **The first load for the ANE is costly:** 0.7–2.5 s, as Core ML compiles the model for the Neural Engine and caches it. Later loads take 20–34 ms. W8A8's 589 ops take longest to compile.
- **Xcode performance reports, CPU and Neural Engine, 2026-11-21:** every version runs all its ops on the ANE (202, or 589 for W8A8).

  | Version | Prediction | Load | Compilation |
  |---|---|---|---|
  | FP16 (Stage 3) | 1.80 ms | 23.03 ms | 95.85 ms |
  | W8 | 1.71 ms | 40.35 ms | 93.09 ms |
  | P6 | 1.69 ms | 38.37 ms | 86.44 ms |
  | W8A8 | 1.66 ms | 50.83 ms | 149.63 ms |

  The order matches the app's benchmark, but the spread is smaller: W8A8 is 8% faster than FP16 here, against 21% in the app. Xcode's report times fewer runs on its own inputs, and the P6 and W8A8 reports were still finishing when captured, so the app's 500 runs on real frames are the numbers to use. Xcode's per-op estimates for W8A8 are about half FP16's (2 µs against 4 µs for the same early ops), which fits 8-bit activations halving the data each op moves. W8A8 also takes longest to compile, at 150 ms, with its 589 ops.
- **Live latency is about 3× the benchmark on the ANE.** In the 11-minute soak, W8 on the ANE took about 5.7 ms per frame, against 1.69 ms back to back. A command-line test (W8, Release, 200 runs each) shows why: every processor gets slower when it sits idle between frames, and the ANE most of all.

  | W8 median | Back to back | 5 ms pause | 33 ms pause (30 fps) |
  |---|---|---|---|
  | ANE | 1.59 ms | 4.93 ms | 5.14 ms (3.2×) |
  | GPU | 6.92 ms | 13.10 ms | 14.12 ms (2.0×) |
  | CPU | 10.97 ms | 11.04 ms | 16.59 ms (1.5×) |

  This fits each processor dropping to a low-power state between frames and paying to wake up. The ANE pays even after a 5 ms gap, while the CPU slows gradually as the gap grows. At camera speed, the ANE is still the fastest. The benchmark's back-to-back numbers are best-case figures, and a live app should be judged at its real frame rate.
- **Soak test (W8, ANE, 679 s, 2026-11-21; `inference/coreml/soak_mac_m2_w8_ane_2026-11-21.csv`):** the thermal state stayed nominal throughout. Per-minute median model time held at 5.6–5.7 ms for 10 minutes, then 6.2–6.4 ms in the last 2 minutes, which could be heat or the Mac being used again at the end. The camera delivered a steady 24 fps, probably lowered for exposure, since each frame took only about 8 ms in total. Resize took about 2.5 ms at 640×480.
- **Power (`powermetrics`, 20 one-second samples, first 2 dropped as start-up; `inference/coreml/power_*.txt`, 2026-11-21).** DepthCam readings are the whole live app (camera, resize, model, display) at about 24 fps. ONNX is `onnx_bench` running flat out at about 25 inferences a second, so the two rates are similar.

  | Reading | CPU | GPU | ANE | Total | Above idle |
  |---|---|---|---|---|---|
  | Idle | 251 mW | 0 | 0 | 251 mW | — |
  | DepthCam FP16, CPU + Neural Engine | 348 | 18 | 326 | **693 mW** | 442 mW |
  | DepthCam W8A8, CPU + Neural Engine | 403 | 28 | 299 | 731 mW | 480 mW |
  | DepthCam FP16, CPU + GPU | 646 | 636 | 260 | 1,542 mW | 1,291 mW |
  | DepthCam FP16, CPU only | 1,552 | 36 | 349 | 1,937 mW | 1,686 mW |
  | ONNX Runtime, 4 threads | 18,293 | 20 | 0 | **18,313 mW** | 18,062 mW |

  - The whole live app on the ANE uses **26× less power than ONNX's model alone** (0.69 W against 18.3 W) at a similar rate. That's about 29 mJ per frame for everything, against about 730 mJ per ONNX inference.
  - On the same app, the ANE setting uses 2.2× less power than the GPU and 2.8× less than CPU only.
  - W8A8 shows no power saving over FP16 on the ANE (731 against 693 mW, within the noise of 20 samples), despite being 21% faster back to back.
  - **Unexplained:** the ANE draws 260–350 mW in every DepthCam reading, even with the model on the GPU or CPU alone, and 0 mW for ONNX with the camera off. Something tied to the camera seems to use the ANE, possibly macOS's camera video effects, but that's unverified. It also means the model's own ANE power can't be separated from that baseline.
  - The first four DepthCam readings were invalid: macOS App Nap throttled DepthCam once the terminal was in front, so power fell to idle after about a second. The app now holds a `ProcessInfo.beginActivity(.userInitiated)` token, and the readings above are the redone ones.
- **Bug fixed:** the device name `Mac14,15` contains a comma, which split it across two CSV columns. The app now quotes it, and the copy in `inference/coreml/` is corrected.

**iPhone 14 results (2026-11-21, iOS 27.2.1, Release build, unplugged, Low Power Mode off, thermal state nominal throughout; `inference/coreml/benchmark_iphone14_2026-11-21.csv`).** p50 in ms, with the ratio to the Mac in brackets:

| Version | CPU + ANE | CPU + GPU | CPU only | First / second load, ANE |
|---|---|---|---|---|
| FP16 | 2.79 (1.54×) | 19.14 (2.38×) | 12.55 (1.13×) | 78* / 25 ms |
| W8 | 2.49 (1.47×) | 19.03 (2.25×) | 12.70 (1.14×) | 1,003 / 32 ms |
| P6 | 2.20 (1.38×) | 19.02 (2.26×) | 12.68 (1.14×) | 942 / 31 ms |
| W8A8 | 2.21 (1.54×) | 18.67 (2.20×) | 20.91 (1.21×) | 109* / 42 ms |

\* The `all` run just before had already compiled the model for the ANE (821 and 1,558 ms), so this first load hit the cache.

- **Placement is identical to the Mac:** 202 of 202 ops on the ANE for every version with CPU + ANE (589 for W8A8, counting the quantize ops).
- **ANE is 1.4–1.5× slower than the M2's,** even though both are 16-core Neural Engines rated about 15.8 TOPS.
- **Weight compression helps more on the iPhone:** FP16 → P6 saves 0.59 ms (21%) there against 0.22 ms (12%) on the Mac. Both points fit the iPhone having less memory bandwidth (LPDDR4X against the M2's LPDDR5), with weight loading a bigger share of the time. That's an inference from the numbers, not measured directly.
- **On the iPhone, the GPU is the slowest choice:** 19 ms, slower than the CPU's 12.6 ms. The iPhone 14's 5-core GPU is 2.2–2.4× slower than this M2's 10-core GPU, while the CPUs are within 15% of each other.
- **W8A8 is 1.7× slower on the CPU than FP16,** as on the Mac, because of the extra quantize and dequantize ops.
- Second loads for the GPU and CPU take about 1 ms on the iPhone against about 20 ms on the Mac. This isn't explained and isn't investigated.

**iPhone soak (W8A8 on CPU + ANE, 601 s, unplugged; `inference/coreml/soak_iphone14_w8a8_ane_2026-11-21.csv`).** The Mac soak used W8, so the versions differ; both run fully on the ANE.
- Thermal state stayed **nominal** for all 10 minutes. Model p50 held at 5.05–5.12 ms every minute (p99 about 5.4 ms), at 24 FPS, with no slowdown. The Mac's W8 drifted from about 5.7 to 6.3 ms over 11 minutes.
- Live latency is 2.3× the benchmark's 2.21 ms, the same idle-between-frames effect seen on the Mac.
- The resize takes about 3.3 ms on the iPhone against about 2.5 ms on the Mac.

**iPhone Xcode performance reports (2026-11-21, compute units All, iOS 27.2.1).** Xcode's report disagrees with what the app measures:

| Version | Xcode median prediction | Xcode load / compile | Xcode placement | App benchmark (`all`) |
|---|---|---|---|---|
| FP16 | 4.76 ms | 536 / 107 ms | 185 ANE, 4 GPU | 2.78 ms, 100% ANE |
| W8 | 4.30 ms | 629 / 111 ms | 185 ANE, 4 GPU | 2.50 ms, 100% ANE |
| P6 | 4.04 ms | 665 / 107 ms | 185 ANE, 4 GPU | 2.21 ms, 100% ANE |
| W8A8 | 143.44 ms | 1,403 / 173 ms | 50 ANE, 530 CPU | 2.20 ms, 100% ANE |

- For FP16, W8 and P6, Xcode puts the four input ops (`cast`, `reverse`, `sub`, `mul`) on the GPU and the rest on the ANE. On the Mac, Xcode put all of them on the ANE. A GPU-to-ANE handoff at the start fits Xcode's times being about 2 ms higher than the app's, but that isn't verified.
- For W8A8, Xcode puts 530 of 580 ops on the CPU, starting with the quantize and dequantize ops, and a prediction takes 143 ms. The app's `MLComputePlan` and timings on the same phone show 100% ANE at 2.2 ms. On the Mac, Xcode's report showed W8A8 on the ANE at 1.66 ms.
- A likely cause is that Xcode's report compiles the `.mlpackage` on the phone, with iOS 27.2.1's compiler. The app bundles a model compiled at build time by Xcode 26.3's compiler. That's unverified, and not investigated further. **The lesson:** check placement in the shipped app on the target device, not only in a tool.

**iPhone power (Instruments Power Profiler, 2026-11-21, 2 min 16 s, phone plugged in; screenshot only, not saved).** About 30 s each of FP16 on CPU + ANE, FP16 on CPU + GPU, FP16 on CPU only, and W8A8 on CPU + ANE. Power Profiler reports a unitless "power impact" per process for the CPU, GPU, display and networking. It has no ANE track and no watts, so this is a relative comparison only.
- **CPU + ANE (FP16 and W8A8):** CPU and GPU impact both near zero, the lowest of the four.
- **CPU + GPU:** a clear, steady GPU impact for the whole section, with little CPU.
- **CPU only:** the highest sustained CPU impact of the run.
- Short CPU spikes at each switch are model loads.
- Thermal state stayed nominal, and brightness was fixed at 40%. Whole-system battery drain reads 0.0%/hr because the charger was connected, so it can't be used.
- **Conclusion:** the iPhone ranks the settings the same way as the Mac's `powermetrics` readings (ANE lowest, then GPU, then CPU). The ANE's own power isn't visible on the iPhone.

**Mac vs iPhone summary (task 33):**
- **Same placement:** 100% ANE in the app on both devices.
- **ANE latency:** 1.4–1.5× slower on the iPhone. Compression saves more there (FP16 → P6: 21% against 12%).
- **GPU and CPU:** the iPhone GPU is 2.2–2.4× slower; the CPUs are within 15%. On the iPhone the GPU is slower than the CPU.
- **Heat:** both stayed nominal through 10-minute soaks. The iPhone held a flat 5.1 ms (W8A8); the Mac drifted from 5.7 to 6.3 ms (W8).
- **Xcode's report on the iPhone disagrees with the app** (input ops on the GPU; W8A8 mostly on the CPU at 143 ms). On the Mac, Xcode and the app agree.

**Comparison with ONNX.** A small C++ runner in `coreml/` times `session.Run` on the same frames with the corrected preprocessing, so model time is compared with model time. Live FPS is reported separately, since the app and the OpenCV pipeline differ in display and threading.

**Xcode performance report steps** are given as exact clicks at Phase 7. In short: double-click the `.mlpackage`, open the **Performance** tab, click **+**, pick the device and compute unit, and run it. A report for the iPhone needs it connected by cable.

**Power and heat:**
- **Mac:** `sudo powermetrics` reports CPU, GPU and ANE power. It asks for your password, so you run it yourself.
- **iPhone:** Instruments' power tools, with the phone connected. Exact availability on your Xcode is confirmed at Phase 8.
- **Soak test:** run the live app for 10 minutes while logging latency and thermal state each second. The results are plotted over time. Your fanless Air makes this a meaningful test, and it runs once on each device for the best variant.

## 6. Swift app (minimal)

**Project.** Called `DepthCam`, in `coreml/DepthCam/`, using Xcode's Multiplatform App template so one codebase runs on Mac and iPhone. You create it with click-by-click steps, and Xcode picks up files I add to the folder automatically.

**Frame flow:**

```
Camera (BGRA CVPixelBuffer, IOSurface-backed)
  → drop if the model is still busy
  → resize to 256×256 with vImage into a reusable buffer pool
  → Core ML prediction
  → Float16 CVPixelBuffer → Metal texture → colormap shader → screen
```

**Stale frames are dropped, not queued.** `alwaysDiscardsLateVideoFrames = true` plus a busy flag means the display shows the newest result. This also fixes the old loop's design, where everything waited in turn.

**Resizing decision: vImage with direct Core ML, not Vision.** Direct use teaches `CVPixelBuffer` handling and lets us time the resize separately from the model. A stretch to 256×256 matches the current pipeline. The app records one resized buffer so we can measure any difference from OpenCV's resize.

**Files:**

| File | What it does |
|---|---|
| `DepthCamApp.swift` | App entry point |
| `ContentView.swift` | Live view, FPS and latency labels, model and compute-unit pickers |
| `CameraManager.swift` | `AVFoundation` capture and frame dropping |
| `ModelRunner.swift` | Loads a variant with chosen compute units and runs predictions |
| `FrameResizer.swift` | vImage resize and reusable buffer pool |
| `DepthRenderer.swift` | Metal view with colormap shader |
| `BenchmarkRunner.swift` | The benchmark mode from section 5 |
| `ComputePlanReport.swift` | The `MLComputePlan` per-op report |
| `Info.plist` setting | Camera permission text, without which the app can't use the camera |

**Model picker.** The app bundles all four variants and loads by name, so you can switch live. Switching takes a moment, because a load may need to compile the model. Load time is displayed.

**Mac first, iPhone second.** The Mac build uses "Sign to Run Locally" with no Apple ID. The iPhone needs a free Apple ID, a cable and Developer Mode, and free signing expires after 7 days, after which you simply reinstall.

**As built (2026-11-05):**
- **Models:** the four `.mlpackage` files are copied into `DepthCam/DepthCam/Models/` (gitignored), and Xcode compiles them into the app.
- **Shader:** compiled when the app starts, from source inside `DepthRenderer.swift`. Xcode 26 ships the offline Metal compiler as a separate ~700 MB download, and this avoids needing it.
- **Resize:** vImage resamples with a Lanczos filter, not bilinear like `cv::resize`, so the app's input differs slightly from the Python scripts'. The benchmark uses frames already resized in Python (`export_bench_frames.py`), so its timings use exactly the frames `evaluate.py` measured.
- **Camera size:** the Mac's FaceTime camera ignores the `.vga640x480` preset and sent 1920×1080, which made the resize take about 7.5 ms and squashed the 16:9 picture. The app now picks the camera's 640×480 format directly, after the session is configured. On the Mac, it also asks the video output for 640×480.
- **Live timings** leave out the first 10 predictions after each load, which are slow while the model warms up. One included warm-up frame had pushed the live p99 to 249 ms.
- **Benchmark mode:** writes `benchmark_<device>_<time>.csv` to the app's Documents folder. On the Mac, that's `~/Library/Containers/com.rohaanadvani.DepthCam/Data/Documents/`.

**Phase 6 check:** the app shows live depth from the FaceTime camera, labels show FPS and latency, and switching variants changes file size and timing.

## 7. Optional stretch

Only after the main results are done, and only if you want them:
- Temporal EMA in a Metal compute shader, using the formula from `DepthPostProcessor`.
- A sphere occlusion demo in the app, reusing the shader logic from the main project.

## 8. Deliverables

| Path | Contents | In git? |
|---|---|---|
| `coreml/PLAN.md`, `RESULTS.md` | This plan and the results | Yes |
| `coreml/requirements.txt` | Pinned Python versions | Yes |
| `coreml/capture_frames.py` | Saves calibration and held-out frames | Yes |
| `coreml/convert.py`, `compress.py`, `evaluate.py` | Conversion, compression and accuracy | Yes |
| `coreml/onnx_baseline/` | C++ ONNX timing runner | Yes |
| `coreml/DepthCam/` | The Xcode project | Yes |
| `coreml/results/` | Summary tables and a few chosen images | Yes |
| `coreml/.venv/`, `third_party/`, `data/`, `models/` | Environment, clones, your frames and generated models | No |

`.gitignore` gets those exclusions. Generated models are reproducible from the scripts, which keeps the repo small. Nothing is committed unless you ask.

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| The pinned setup won't install or import | Adjust within coremltools' supported versions and record the change |
| MiDaS imports break with a newer `timm` | Pin an older `timm`, or copy only the small-model classes |
| Backbone download via `torch.hub` fails | Use the pinned local clone |
| Tracing or conversion error on an op | Rewrite that op in the wrapper, then repeat the numerical check |
| Core ML output doesn't match PyTorch | Stop and debug layer by layer, starting with normalization |
| Ops don't land on the ANE | Run the section 2 experiments, and report honestly if some stay on CPU or GPU |
| ANE isn't faster than GPU | A valid result. Report latency, power and placement together |
| W8A8 accuracy drops sharply | Try different calibration frames and per-op exclusions, then report the result |
| Image output is awkward | Use `MLMultiArray` output and accept a copy |
| iPhone signing or Developer Mode problems | Step-by-step fixes in Phase 8. Mac results stand regardless |
| A real app hits an iOS 27 problem on Xcode 26.3 that the blank test app didn't | Upgrade to macOS Tahoe 26.6 and Xcode 27, and rerun any Mac benchmarks taken before the upgrade |
| Heat skews benchmarks on the fanless Air | Cooldowns, logged thermal state, and a rerun of anything taken while hot |
| 8 GB of memory runs short during conversion | Close other apps and convert one variant at a time |

## 10. Final output: `RESULTS.md`

- **Results table, one per device:** variant | size (MB) | load time | latency p50/p90/p99 per compute unit | % ops on ANE by count and cost | accuracy vs baseline | live FPS.
- **Why each variant performed as it did**, tied to what we measured. Ideas like "palettization reduces memory traffic" appear only if our data supports them.
- **What ran on the ANE and what didn't**, with reasons.
- **Comparison with the ONNX pipeline**, covering model time and live FPS separately.
- **Mac vs iPhone comparison**, plus heat and power findings.
- **Five interview talking points** using only measured numbers.
