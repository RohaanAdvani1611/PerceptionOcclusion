# MiDaS on Core ML: implementation plan

**Status: awaiting your approval. No code, installs or project changes happen until you approve.**

**Goal:** convert MiDaS v2.1 small to Core ML, compare four compressed variants on size, latency, accuracy and compute-unit placement, and run them on live camera frames in a small Swift app, on the M2 Mac first and then the iPhone 14.

All new work stays in `coreml/`. The only existing file this plan changes is `.gitignore`.

## Facts this plan is built on

| Item | Value | How it affects the plan |
|---|---|---|
| Mac | MacBook Air M2 (`Mac14,15`), 8 GB, macOS 15.6.1 | It has no fan, so long runs will likely slow down from heat. Memory is tight, so close other apps while benchmarking. |
| iPhone | iPhone 14 (A15), iOS version not yet confirmed | iOS 17.4 or later is needed for per-op placement data. |
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
| 3 | Environment, MiDaS source, conversion, numerical check | 2–4 hours |
| 4 | ANE placement analysis | 1–3 hours |
| 5 | Capture frames, compress, evaluate accuracy | 2–3 hours |
| 6 | Swift app on Mac, with live view and benchmark mode | 3–5 hours |
| 7 | Mac benchmarks, performance reports, soak test | 2–3 hours |
| 8 | Deploy to iPhone 14 and compare | 2–4 hours |
| 9 | `RESULTS.md` and talking points | 1–2 hours |
| Optional | Temporal filter or occlusion demo in the app | 2–4 hours |

Most of the uncertainty is in Phase 4 and in signing during Phase 8.

## 1. Conversion

**Setup.** Create `coreml/.venv` with Python 3.12 and pinned versions: `torch==2.7.*`, `coremltools==9.0`, and the other packages listed in `coreml/requirements.txt`.

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
| PyTorch vs corrected ONNX | Report the values | Measures the `align_corners` gap from Phase 0 |

These thresholds are starting targets I chose, not published limits. If the first comparison fails, nothing else proceeds.

**Phase 3 check:** the `.mlpackage` opens in Xcode showing the expected inputs and outputs, the script prints passing values, and a side-by-side depth image looks right.

## 2. Neural Engine fit

**How we find out where ops run.** We use three sources, because no single one gives the full picture:

1. **`MLComputePlan`** (a Swift tool) lists each op's preferred and supported devices and its estimated cost. This produces the "% on ANE" figure.
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

## 3. Compression variants

Everything is built from the same FP16 model using `coremltools.optimize.coreml`. Sizes are my estimates from the 16.7M parameters. Real numbers replace them in `RESULTS.md`.

| Variant | How | Data needed | Estimated size |
|---|---|---|---|
| Baseline FP16 | `compute_precision=FLOAT16` | None | ~33 MB |
| 8-bit weights | `linear_quantize_weights` | None | ~17 MB |
| 6-bit palettized | `palettize_weights`, `nbits=6`, `mode="kmeans"` | None | ~13 MB |
| W8A8 | `linear_quantize_activations`, then `linear_quantize_weights` | 100–200 frames | ~17 MB |

**What to expect on your hardware.** Coremltools' docs say int8-int8 ANE compute arrives from iPhone 15 Pro onwards, so neither the M2 nor the A15 has it. W8A8 is an experiment. It may run with no speedup, or even more slowly, and it may lose accuracy. Any of those is a useful result if we can explain it.

**Calibration frames.** `coreml/capture_frames.py` saves frames from your webcam, from a few different scenes and lighting conditions.
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
| iPhone is below iOS 17.4 | Update it, or rely on Xcode's report and latency for placement |
| Heat skews benchmarks on the fanless Air | Cooldowns, logged thermal state, and a rerun of anything taken while hot |
| 8 GB of memory runs short during conversion | Close other apps and convert one variant at a time |

## 10. Final output: `RESULTS.md`

- **Results table, one per device:** variant | size (MB) | load time | latency p50/p90/p99 per compute unit | % ops on ANE by count and cost | accuracy vs baseline | live FPS.
- **Why each variant performed as it did**, tied to what we measured. Ideas like "palettization reduces memory traffic" appear only if our data supports them.
- **What ran on the ANE and what didn't**, with reasons.
- **Comparison with the ONNX pipeline**, covering model time and live FPS separately.
- **Mac vs iPhone comparison**, plus heat and power findings.
- **Five interview talking points** using only measured numbers.
