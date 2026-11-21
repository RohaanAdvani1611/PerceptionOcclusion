# Real-Time Perceptual Occlusion Engine for AR Systems

<p align="center">
  <a href="docs/demo.mp4"><img src="docs/demo.gif" alt="Perceptual Occlusion Engine demo: a virtual sphere slides behind a real laptop, cup and flowers" width="100%"></a>
  <br>
  <sub><a href="docs/demo.mp4">▶ Watch the full video with sound</a></sub>
</p>

## 1. Overview

This project implements a real-time perception and rendering pipeline that enables occlusion of virtual objects by real-world geometry using a single monocular camera.

The system estimates scene depth from RGB input with a neural network, renders a virtual object on the GPU with Metal, and composites the two so that the virtual object is hidden wherever a real object is in front of it. The goal is to improve visual realism in AR applications by enforcing proper depth ordering between real and virtual elements.

The `coreml/` extension moved the depth model to Apple's Core ML and Neural Engine, and the engine now uses it by default. Depth inference fell from 44.6 ms to 5.5 ms per frame, which doubled the frame rate to about 23 FPS and cut processing latency from 65 ms to 27 ms. The same model also runs live on an iPhone 14 in a Swift app. See [Section 9](#9-core-ml-and-neural-engine-extension-coreml).

## 2. System Requirements & Setup

### Platform

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | macOS 12.0 Monterey | macOS 13.0+ Ventura |
| Hardware | Apple Silicon (M1 or later) | — |
| Camera | Any USB or built-in webcam | — |

Metal is Apple's GPU API. The renderer requires Apple Silicon because the CPU and GPU share memory directly through its unified memory: the CPU writes the camera and depth inputs straight into GPU-visible buffers and reads the output without a copy.

### Dependencies

```bash
# Xcode Command Line Tools (required for clang++ and Metal runtime shader compilation)
xcode-select --install

# Package manager
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Build tools and libraries
brew install cmake ninja opencv onnxruntime
```

### Model

By default the engine runs MiDaS v2.1 small through Core ML on the Neural Engine, using the 8-bit-weight model at `coreml/models/midas_small_w8.mlpackage`. That file isn't in git; build it with the steps in Section 9. If it's missing, the engine falls back to the ONNX model at `assets/models/midas.onnx` (64 MB) on the CPU. If that file is missing too, download it:

```bash
curl -L https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx \
     -o assets/models/midas.onnx
```

### Build & Run

```bash
./run.sh       # configures, builds into build/ and launches ./engine
```

Or manually, with an optimized build:

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build-release
./build-release/engine
```

The engine opens camera index `0` (see `Engine.cpp`); change it if your webcam is on a different index. Run `./build-release/engine --depth onnx` to use the old ONNX path instead of Core ML. The AR View shows which depth model is running. The first launch with Core ML takes about 1.5 s while it compiles the model for the Neural Engine.

### Controls

| Key | Action |
|---|---|
| U / D | Move sphere up / down |
| L / R | Move sphere left / right |
| F / B | Move sphere forward / back (changes what hides it) |
| `[` / `]` | Decrease / increase occlusion threshold (calibration, see 5.4) |
| ESC, Q, Ctrl+C | Quit |

## 3. Motivation

Most lightweight AR systems render virtual objects directly on top of camera input without understanding scene geometry, resulting in visually incorrect compositions.

This project addresses that limitation by introducing:

- Real-time depth perception
- Depth-aware compositing
- Stability-focused depth filtering

## 4. System Architecture

The pipeline runs as a single synchronous loop, one stage after another per frame:

```
Camera → DepthEstimator → DepthPostProcessor → Renderer (Metal) → OcclusionPass → Display
```

### Input Stage

- Captures live 640×480 RGB frames from a monocular camera via OpenCV (AVFoundation backend)

### Perception Stage

- Runs MiDaS v2.1 small through Core ML on the Neural Engine (8-bit weights) to produce a per-pixel relative depth map. ONNX Runtime on the CPU (4 threads) remains available with `--depth onnx`
- Normalizes depth to [0, 1] using 1st/99th percentile clipping, so outlier pixels don't collapse the scale

### Depth Processing Stage

- Bilateral filter for edge-preserving spatial denoising, with sigmas set for depth values in [0, 1]
- Per-pixel adaptive temporal smoothing (EMA) that responds faster where depth changes a lot, to avoid ghosting on moving objects

### Rendering Stage

- Draws the camera frame and a lit virtual sphere with Metal into one offscreen BGRA target
- Derives the projection from the camera intrinsics so the virtual and real perspectives match

### Compositing Stage

- Closes small holes in the real depth map on the CPU
- The sphere's fragment shader compares its distance with the real depth at each pixel and discards pixels where the real surface is closer, so compositing happens on the GPU

## 5. Core Components

### 5.1 Monocular Depth Estimation (`perception/DepthEstimator`)

A lightweight neural network infers scene depth from a single RGB frame, so no stereo camera or LiDAR is needed. MiDaS outputs *relative* (not metric) depth, normalized per frame to [0, 1] where 1 is closest.

Two implementations share one interface (`perception/DepthModel`):

- `CoreMLDepthEstimator` (default) resizes the frame to 256×256 into a reused IOSurface-backed pixel buffer, runs the Core ML model on the CPU and Neural Engine, and reads back its Float16 depth image. The model normalizes its own input.
- `DepthEstimator` runs the ONNX model with ONNX Runtime on the CPU.

Both upsample to the frame size and apply the same percentile normalization.

Key considerations:

- Model latency vs accuracy tradeoff
- Resolution scaling for real-time performance

### 5.2 Depth Refinement and Temporal Stability (`perception/DepthPostProcessor`)

- Edge-aware spatial filtering (bilateral)
- Temporal smoothing with a per-pixel adaptive weight: static regions are smoothed heavily, regions with large depth change update quickly

This reduces flickering and keeps occlusion edges stable.

### 5.3 GPU Rendering (`rendering/MetalContext`, `rendering/Renderer`)

- Compiles the Metal shaders at startup and builds a procedural sphere mesh
- Each frame, draws the camera frame as the background, then the sphere with diffuse lighting and a depth buffer
- The output is a texture backed by a shared buffer, which OpenCV wraps with no readback copy. Depth is written into a shared buffer in place; the camera frame needs one CPU conversion from BGR to BGRA, and `imshow` makes its own copy for the window

### 5.4 Depth-Aware Compositing (`compositing/OcclusionPass`)

- Applies a morphological close to the smoothed depth map on the CPU to fill small holes and prevent flickery punch-through at edges
- For each sphere pixel, the fragment shader converts its distance to the same scale as the real depth with `sphereDepth = threshold × 1.5 / distance`, and discards the pixel if `realDepth > sphereDepth`
- Moving the sphere closer or farther therefore changes what hides it, and the sphere can pass partly through a surface

MiDaS depth is relative and renormalized every frame, so this is a calibrated approximation, not a metric comparison. The threshold, adjustable with `[` and `]`, sets the real depth value treated as the sphere's starting distance of 1.5 m. The model ignores the unknown shift in MiDaS's scale-and-shift ambiguity, and the sphere's apparent placement can vary as the per-frame normalization changes.

## 6. Performance & Benchmarking

### Measuring performance

The engine profiles every frame. Each stage is timed separately, and the first 30 frames are excluded as warm-up (ONNX Runtime and Metal are slow on their first frames).

```bash
./bench.sh                  # Release build, 600 measured frames → inference/bench_results.csv
./bench.sh 1200 run2.csv    # custom frame count and file name, saved in inference/
./bench.sh 1200 onnx.csv --depth onnx   # extra flags go to the engine
```

The run stops on its own after the requested number of frames (or press Q / ESC to stop early) and prints a summary table. The engine also accepts the flags directly:

```bash
./engine --bench 600 --csv ../inference/results.csv
```

| Stage | What it measures |
|---|---|
| `read` | Waiting for and grabbing the next camera frame |
| `infer` | MiDaS preprocessing, inference and normalization |
| `post` | Bilateral filter + temporal smoothing |
| `render` | Camera frame conversion and full Metal render including wait |
| `gpu` | GPU execution time only (from the Metal command buffer) |
| `occlusion` | Morphological close of the depth map |
| `display` | Drawing the overlay text and presenting the windows |
| `latency` | Processing latency: camera frame handed to the engine → frame on screen |

`latency` does not include camera sensor exposure, driver buffering or display refresh. To measure **glass-to-glass** latency, point the camera at a phone running a millisecond stopwatch, place the AR window next to the phone, photograph both and subtract the two readings (average ~10 photos).

Always benchmark a Release build; `run.sh` builds Debug.

### Measured results

Apple M2 (8 GB), macOS 15.6, OpenCV 5.0, 640×480 input, built-in FaceTime HD camera, occlusion threshold 0.5. Both runs have 1,200 frames, recorded back to back on 2026-11-21 (`inference/coreml_w8.csv` and `inference/onnx_rerun.csv`).

**Core ML, W8 model on the Neural Engine (default):**

| Stage | Mean (ms) | p50 | p95 | p99 |
|---|---|---|---|---|
| read | 16.3 | 15.0 | 25.2 | 31.4 |
| **infer** | **5.5** | **5.7** | **6.7** | **7.2** |
| post | 5.9 | 5.9 | 6.8 | 7.1 |
| render | 0.5 | 0.5 | 0.5 | 0.6 |
| gpu | 0.06 | 0.06 | 0.06 | 0.07 |
| occlusion | 0.9 | 0.9 | 1.0 | 1.1 |
| display | 14.2 | 15.1 | 15.7 | 16.0 |
| **latency** | **27.0** | **28.2** | **29.9** | **31.0** |

**ONNX Runtime 1.31 on the CPU (`--depth onnx`):**

| Stage | Mean (ms) | p50 | p95 | p99 |
|---|---|---|---|---|
| read | 22.3 | 22.6 | 28.3 | 38.1 |
| **infer** | **44.6** | **43.6** | **50.9** | **60.9** |
| post | 5.3 | 5.2 | 6.3 | 8.1 |
| render | 0.6 | 0.5 | 0.6 | 0.7 |
| gpu | 0.06 | 0.06 | 0.07 | 0.07 |
| occlusion | 0.5 | 0.5 | 0.5 | 0.9 |
| display | 13.6 | 13.9 | 14.2 | 15.1 |
| **latency** | **64.6** | **63.8** | **71.8** | **83.3** |

| | ONNX, CPU | Core ML, Neural Engine | Change |
|---|---|---|---|
| Depth inference (mean) | 44.6 ms | 5.5 ms | 8.1× faster |
| Processing latency (mean / p99) | 64.6 / 83.3 ms | 27.0 / 31.0 ms | 2.4× lower |
| Loop time, sum of stages | 86.9 ms (~11.5 FPS) | 43.3 ms (~23 FPS) | 2× the frame rate |

- **With ONNX, depth inference was the bottleneck,** at 69% of processing latency.
- **With Core ML it's 20% of processing latency.** The bottleneck has moved to the camera wait (`read`, 16 ms) and the window display (`display`, 14 ms), which together are 70% of the loop. Rendering and compositing together still cost about 1 ms.
- **Next step for 30 FPS:** move capture and display off the processing loop (see Future Work).
- **Live vs benchmark:** inference in the live engine (5.5 ms) is slower than back to back on saved frames (3.7 ms), because the Neural Engine powers down between frames. `coreml/RESULTS.md` covers this effect.
- **Accuracy:** on 100 saved frames, the Core ML and ONNX depth maps correlate at 0.971, with 6.8% of mask pixels flipping at 0.5. This matches the 6.7% gap measured between ONNX and the original PyTorch model, which comes from the ONNX path's resize. So the Core ML depth is the closer of the two to the original model.

## 7. Challenges and Solutions

- Noisy depth → bilateral filtering and percentile-clipped normalization
- Flickering → adaptive temporal smoothing and morphological cleanup of the occlusion mask
- Ghosting on moving objects → per-pixel adaptive smoothing weight
- Relative (non-metric) depth → per-pixel comparison in inverse depth with a tunable calibration
- Alignment between CV and rendering → consistent camera intrinsics and coordinate systems across all stages

## 8. Results

- Real-time occlusion of a virtual object by real-world geometry from a single webcam
- Stable occlusion edges thanks to spatial and temporal filtering
- ~23 FPS and ~27 ms processing latency on an Apple M2 with Core ML on the Neural Engine, up from ~11.5 FPS and ~65 ms with ONNX on the CPU (see Section 6)

## 9. Core ML and Neural Engine extension (`coreml/`)

The C++ engine spends about 69% of its processing time on depth inference (Section 6). This extension asks how fast, small and accurate the same MiDaS model can be with Apple's on-device ML stack. It also asks which processor each part of the model runs on. The work lives in `coreml/`, and the resulting W8 model now powers the engine (step 9 below).

The full write-up, including why each result happened and five talking points, is in [coreml/RESULTS.md](coreml/RESULTS.md). The detailed plan and every side experiment are in [coreml/PLAN.md](coreml/PLAN.md).

### What was done

1. **Prepare.**
   - Set up a pinned Python 3.12 environment with PyTorch 2.7.0 and coremltools 9.0.
   - Downloaded the official MiDaS source and weights.
   - Captured 150 calibration frames and 100 held-out test frames from real rooms.
2. **Convert.**
   - Wrapped MiDaS v2.1 small so normalization happens exactly once, inside the model.
   - Converted it to a Core ML ML Program, with a camera image as input and a depth image as output.
   - Checked the result against PyTorch: Pearson 1.000000 for FP32, and 0.99998 for FP16 on the Neural Engine.
3. **Neural Engine analysis.**
   - Listed which processor runs each of the model's 202 operations, using `MLComputePlan` and Xcode's performance report.
   - Two ops ran on the CPU because Core ML scaled the input image there. Folding that scale into the model put 202 of 202 ops on the Neural Engine.
4. **Compress and evaluate.**
   - Built three smaller versions: 8-bit weights (W8), 6-bit palettized weights (P6), and 8-bit weights plus 8-bit activations (W8A8).
   - Measured accuracy on the held-out frames, including how many pixels flip between "near" and "far" in the occlusion mask.
5. **Swift app (`coreml/DepthCam/`).** DepthCam is a SwiftUI app for macOS and iOS:
   - **Pipeline:** camera frames are resized with vImage, depth comes from Core ML, and Metal displays it with a colour map.
   - **Controls:** pickers switch the model version and the compute units live.
   - **Measurement:** labels show live latency and FPS. A benchmark mode times all 16 combinations, and a soak log records timings and heat over time.
6. **Mac benchmarks.**
   - Timed 500 predictions per combination, plus a C++ ONNX Runtime timer on the same frames for a fair comparison.
   - Ran Xcode performance reports.
   - Measured power with `powermetrics`, and ran a 10-minute soak test.
7. **iPhone 14.** Installed DepthCam on the phone and repeated the benchmarks, Xcode reports, power check (Instruments Power Profiler) and soak test.
8. **Write up.** [coreml/RESULTS.md](coreml/RESULTS.md).
9. **Back into the engine.**
   - Added `CoreMLDepthEstimator` to the C++ engine, behind the same interface as the ONNX estimator, and made it the default.
   - Depth inference fell from 44.6 to 5.5 ms per frame, and the engine went from about 11.5 to about 23 FPS (Section 6).

### Results

| Version | Size | Mask flips (mean) | Mac M2, Neural Engine | iPhone 14, Neural Engine |
|---|---|---|---|---|
| FP16 | 33.3 MB | 0.17% | 1.81 ms | 2.79 ms |
| W8 | 16.9 MB | 1.05% | 1.69 ms | 2.49 ms |
| P6 | 12.7 MB | 4.58% | 1.59 ms | 2.20 ms |
| W8A8 | 17.0 MB | 8.91% | 1.43 ms | 2.21 ms |
| ONNX Runtime, CPU (this engine's model) | 64 MB | 6.73% | 39.69 ms (CPU) | — |

- **Placement:** every version runs 100% on the Neural Engine in the app, on both devices.
- **Speed:** Core ML FP16 on the Neural Engine is **22× faster** than the ONNX model on the CPU. Even on the CPU alone, Core ML FP16 is 3.6× faster (11.08 ms).
- **Power:** the whole DepthCam app on the Neural Engine draws **0.69 W** on the Mac, against 1.54 W on the GPU, 1.94 W on the CPU only, and **18.3 W** for the ONNX model alone.
- **Compression:** smaller versions are a bit faster on the Neural Engine but cost accuracy. **W8 is the best trade-off,** at half the size of FP16 with about 1% mask flips. W8A8 isn't usable on these chips: its worst frame flips 87% of the mask.
- **Neural Engine-specific bug:** with every weight in 8-bit, 48 of 100 frames came out as `inf` or `NaN` on the Neural Engine only. Keeping the two output-head convolutions in FP16 fixed it.
- **Live vs benchmark:** at camera speed, processors power down between frames, so live latency is about 3× the back-to-back benchmark. For W8 on the Mac's Neural Engine, that's 5.14 ms against 1.59 ms.
- **Heat:** both devices stayed at "nominal" thermal state through 10-minute runs.

### Running it

1. **Python side (conversion, compression, accuracy):**
   - Set up `coreml/.venv` and the MiDaS downloads with the commands in [coreml/VERSIONS.md](coreml/VERSIONS.md).
   - Capture frames with `python capture_frames.py calibration <scene>` and `python capture_frames.py heldout <scene>`.
   - Then run, from `coreml/`:
     ```bash
     python convert.py              # FP32 and FP16 .mlpackage files in coreml/models/
     python check_conversion.py     # Core ML vs PyTorch vs ONNX
     python analyze_placement.py    # per-op placement tables
     python compress.py             # W8, P6 and W8A8 versions
     python evaluate.py             # accuracy table in coreml/results/accuracy.csv
     python export_bench_frames.py  # frames for the app's benchmark mode
     ```
2. **The app:**
   - Copy the four `.mlpackage` files from `coreml/models/` into `coreml/DepthCam/DepthCam/Models/`.
   - Open `coreml/DepthCam/DepthCam.xcodeproj` in Xcode 26.3 or later.
   - Pick **My Mac** or a connected iPhone as the run destination and press ⌘R. An iPhone needs a free Apple ID set as the signing team, and Developer Mode turned on.
3. **ONNX comparison timer:**
   ```bash
   cmake -S coreml/onnx_baseline -B coreml/onnx_baseline/build && cmake --build coreml/onnx_baseline/build
   ```

Captured frames, models and result images are not tracked by git, because the frames show a private home.

## 10. Future Work

- **Performance:** asynchronous pipeline (camera, inference and rendering on separate threads); camera capture and display now take 70% of each frame, so moving them to their own threads is the main step to ≥ 30 FPS; a Metal window instead of OpenCV's `imshow`
- **Metric depth:** estimate scale and shift so the sphere's distance can be compared with real depth without manual calibration
- **Confidence-aware rendering:** soft blending instead of hard masking in regions where the depth estimate is uncertain
- Semantic segmentation to sharpen object boundaries
- Multi-view reconstruction
- Gaze-based rendering
- Full occlusion on iPhone: DepthCam already runs live depth on an iPhone 14, but doesn't yet render the virtual sphere
- Improved robustness in dynamic scenes

## 11. Key Takeaways

- CV + graphics integration
- Real-time system design and measurement
- Perceptual rendering principles
- On-device ML deployment: Core ML conversion, compression trade-offs, and checking where each operation runs on the Neural Engine, GPU and CPU
