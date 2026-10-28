# Real-Time Perceptual Occlusion Engine for AR Systems

<p align="center">
  <a href="docs/demo.mp4"><img src="docs/demo.gif" alt="Perceptual Occlusion Engine demo: a virtual sphere slides behind a real laptop, cup and flowers" width="100%"></a>
  <br>
  <sub><a href="docs/demo.mp4">▶ Watch the full video with sound</a></sub>
</p>

## 1. Overview

This project implements a real-time perception and rendering pipeline that enables occlusion of virtual objects by real-world geometry using a single monocular camera.

The system estimates scene depth from RGB input with a neural network, renders a virtual object on the GPU with Metal, and composites the two so that the virtual object is hidden wherever a real object is in front of it. The goal is to improve visual realism in AR applications by enforcing proper depth ordering between real and virtual elements.

## 2. System Requirements & Setup

### Platform

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | macOS 12.0 Monterey | macOS 13.0+ Ventura |
| Hardware | Any Mac with Metal support (2012+) | Apple Silicon (M1/M2/M3/M4) |
| Camera | Any USB or built-in webcam | — |

Metal is Apple's GPU API and is supported on all Macs from 2012 onward running macOS 10.11+. The project runs on **both Intel and Apple Silicon** Macs. Apple Silicon is preferred because its unified memory architecture allows zero-copy CPU↔GPU texture transfers; on Intel, the same data crosses the PCIe bus on every frame.

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

The engine uses MiDaS v2.1 small (64 MB) at `assets/models/midas.onnx`. If the file is missing, download it:

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

The engine opens camera index `1` (see `Engine.cpp`); change it if your webcam is on a different index.

### Controls

| Key | Action |
|---|---|
| U / D | Move sphere up / down |
| L / R | Move sphere left / right |
| F / B | Move sphere forward / back |
| `[` / `]` | Decrease / increase occlusion threshold |
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

- Runs MiDaS v2.1 small through ONNX Runtime (CPU, 4 threads) to produce a per-pixel relative depth map
- Normalizes depth to [0, 1] using 1st/99th percentile clipping, so outlier pixels don't collapse the scale

### Depth Processing Stage

- Bilateral filter for edge-preserving spatial denoising
- Per-pixel adaptive temporal smoothing (EMA) that responds faster where depth changes a lot, to avoid ghosting on moving objects

### Rendering Stage

- Renders a lit virtual sphere with Metal into an offscreen BGRA texture with its own depth buffer
- Derives the projection from the camera intrinsics so the virtual and real perspectives match

### Compositing Stage

- Builds an occlusion mask from the real depth map and removes the sphere wherever a real object is in front
- Overlays the remaining sphere pixels on the camera frame and displays the AR view

## 5. Core Components

### 5.1 Monocular Depth Estimation (`perception/DepthEstimator`)

A lightweight neural network infers scene depth from a single RGB frame, so no stereo camera or LiDAR is needed. MiDaS outputs *relative* (not metric) depth, normalized per frame to [0, 1] where 1 is closest.

Key considerations:

- Model latency vs accuracy tradeoff
- Resolution scaling for real-time performance

### 5.2 Depth Refinement and Temporal Stability (`perception/DepthPostProcessor`)

- Edge-aware spatial filtering (bilateral)
- Temporal smoothing with a per-pixel adaptive weight: static regions are smoothed heavily, regions with large depth change update quickly

This reduces flickering and keeps occlusion edges stable.

### 5.3 GPU Rendering (`rendering/MetalContext`, `rendering/Renderer`)

- Compiles the Metal shaders at startup and builds a procedural sphere mesh
- Each frame, renders the sphere with diffuse lighting into an offscreen color texture (transparent background) and depth texture
- Reads the color texture back to the CPU as an OpenCV image; on Apple Silicon this uses shared memory

### 5.4 Depth-Aware Compositing (`compositing/OcclusionPass`)

- Thresholds the smoothed real depth map: pixels closer than the threshold are treated as foreground
- Applies a morphological close to fill small holes and prevent flickery punch-through at edges
- Sets the sphere's alpha to zero wherever the foreground mask is set
- The threshold is adjustable at runtime with `[` and `]`

Because MiDaS depth is relative rather than metric, occlusion uses a tunable threshold instead of a direct comparison with the sphere's depth.

## 6. Performance & Benchmarking

### Measuring performance

The engine profiles every frame. Each stage is timed separately, and the first 30 frames are excluded as warm-up (ONNX Runtime and Metal are slow on their first frames).

```bash
./bench.sh                  # Release build, 600 measured frames → bench_results.csv
./bench.sh 1200 run2.csv    # custom frame count and output file
```

The run stops on its own after the requested number of frames (or press Q / ESC to stop early) and prints a summary table. The engine also accepts the flags directly:

```bash
./engine --bench 600 --csv results.csv
```

| Stage | What it measures |
|---|---|
| `read` | Waiting for and grabbing the next camera frame |
| `infer` | MiDaS preprocessing, inference and normalization |
| `post` | Bilateral filter + temporal smoothing |
| `render` | Full Metal render including wait and texture readback |
| `gpu` | GPU execution time only (from the Metal command buffer) |
| `occlusion` | Building and applying the occlusion mask |
| `display` | Building the AR view and presenting the windows |
| `latency` | Processing latency: camera frame handed to the engine → frame on screen |

`latency` does not include camera sensor exposure, driver buffering or display refresh. To measure **glass-to-glass** latency, point the camera at a phone running a millisecond stopwatch, place the AR window next to the phone, photograph both and subtract the two readings (average ~10 photos).

Always benchmark a Release build; `run.sh` builds Debug.

### Measured results

Apple M2 (8 GB), macOS 15.6, ONNX Runtime 1.31 (CPU), OpenCV 5.0, 640×480 input. Two runs (600 and 1,200 frames) agreed within 0.5 ms; values below are from the 1,200-frame run.

| Stage | Mean (ms) | p50 | p95 | p99 |
|---|---|---|---|---|
| read | 21.9 | 22.3 | 26.9 | 28.5 |
| **infer** | **44.1** | **43.5** | **48.1** | **55.5** |
| post | 5.3 | 5.2 | 6.2 | 6.9 |
| render | 0.7 | 0.7 | 0.8 | 0.9 |
| gpu | 0.05 | 0.05 | 0.05 | 0.05 |
| occlusion | 0.4 | 0.4 | 0.6 | 0.8 |
| display | 14.2 | 14.2 | 14.4 | 14.6 |
| **latency** | **64.6** | **64.1** | **69.0** | **76.5** |

- **Frame rate: ~11.6 FPS** (86.5 ms per loop)
- **Processing latency: ~65 ms** mean, 77 ms p99
- **Depth inference is the bottleneck** at ~68% of processing latency. Rendering and compositing together cost ~1 ms.
- About 22 ms per frame is spent waiting on the camera, because capture runs in the same loop as processing. Moving capture to its own thread would raise the ceiling to ~15 FPS.

## 7. Challenges and Solutions

- Noisy depth → bilateral filtering and percentile-clipped normalization
- Flickering → adaptive temporal smoothing and morphological cleanup of the occlusion mask
- Ghosting on moving objects → per-pixel adaptive smoothing weight
- Relative (non-metric) depth → tunable occlusion threshold
- Alignment between CV and rendering → consistent camera intrinsics and coordinate systems across all stages

## 8. Results

- Real-time occlusion of a virtual object by real-world geometry from a single webcam
- Stable occlusion edges thanks to spatial and temporal filtering
- ~11.6 FPS and ~65 ms processing latency on an Apple M2 (see Section 6)

## 9. Future Work

- **Performance:** asynchronous pipeline (camera, inference and rendering on separate threads); ONNX Runtime CoreML execution provider to run inference on the Neural Engine/GPU; lower inference resolution or running depth every other frame; target ≥ 30 FPS
- **True depth comparison:** compare real depth against the sphere's depth per pixel in a Metal fragment shader, so occlusion responds to the sphere's distance
- **Confidence-aware rendering:** soft blending instead of hard masking in regions where the depth estimate is uncertain
- Semantic segmentation to sharpen object boundaries
- Multi-view reconstruction
- Gaze-based rendering
- Mobile deployment
- Improved robustness in dynamic scenes

## 10. Key Takeaways

- CV + graphics integration
- Real-time system design and measurement
- Perceptual rendering principles
