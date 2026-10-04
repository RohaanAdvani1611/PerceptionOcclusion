# Real-Time Perceptual Occlusion Engine for AR Systems

<p align="center">
  <a href="docs/demo.mp4"><img src="docs/demo.gif" alt="Perceptual Occlusion Engine demo: a virtual sphere slides behind a real laptop, cup and flowers" width="100%"></a>
  <br>
  <sub><a href="docs/demo.mp4">▶ Watch the full video with sound</a></sub>
</p>

## 1. Overview

This project implements a real-time perception and rendering pipeline that enables physically consistent occlusion of virtual objects by real-world geometry using a monocular camera.

The system estimates scene depth from RGB input and integrates it with a GPU-based rendering pipeline to ensure that virtual objects are correctly hidden when obstructed by real-world objects. The goal is to improve visual realism in AR applications by enforcing proper depth ordering between real and virtual elements.

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
brew install cmake opencv onnxruntime
```

### Model

MiDaS v2.1 small must be downloaded separately (64 MB) and placed at `assets/models/midas.onnx`:

```bash
curl -L https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx \
     -o assets/models/midas.onnx
```

### Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
ninja          # or: make -j$(sysctl -n hw.ncpu)
./engine
```

Press **ESC**, **Q**, or **Ctrl+C** to quit.

## 3. Motivation

Most lightweight AR systems render virtual objects directly on top of camera input without understanding scene geometry, resulting in visually incorrect compositions.

This project addresses that limitation by introducing:

- Real-time depth perception
- Depth-aware compositing
- Stability-focused rendering

## 4. System Architecture

The system is structured as a multi-stage pipeline:

### Input Stage

- Captures live RGB frames from a monocular camera
- Preprocesses frames for inference and rendering

### Perception Stage

- Runs a monocular depth estimation model to generate per-pixel depth maps
- Optionally performs semantic segmentation to improve object boundaries

### Depth Processing Stage

- Applies spatial filtering to reduce noise
- Applies temporal smoothing to maintain frame-to-frame consistency
- Generates a refined and stable depth map suitable for rendering

### Rendering Stage

- Renders virtual 3D objects using GPU acceleration
- Maintains a virtual depth buffer aligned with camera coordinates

### Compositing Stage

- Performs per-pixel depth comparison between real and virtual geometry
- Applies occlusion masking to correctly hide virtual objects
- Outputs the final composited AR frame

## 5. Core Components

### 5.1 Monocular Depth Estimation

A lightweight neural network is used to infer scene depth from a single RGB frame. This enables deployment without specialized hardware such as stereo cameras or LiDAR.

Key considerations:

- Model latency vs accuracy tradeoff
- Resolution scaling for real-time performance

### 5.2 Depth Refinement and Temporal Stability

- Temporal smoothing
- Edge-aware spatial filtering
- Optional motion compensation  
  This reduces flickering and ensures stable occlusion edges.

### 5.3 GPU-Based Occlusion (Metal Rendering Pipeline)

The rendering and compositing pipeline is implemented using Metal.

- Render virtual geometry
- Maintain depth buffer
- Perform fragment shader depth comparisons  
  This allows efficient, per-pixel occlusion decisions at real-time frame rates.

### 5.4 Depth-Aware Compositing

Per-pixel comparison between real and virtual depth to determine visibility. Virtual fragments are masked when occluded by closer real-world geometry, producing physically consistent results.

### 5.5 Confidence-Aware Rendering

To handle uncertainty in depth predictions, the system incorporates a confidence-aware blending strategy:

- Estimates reliability of depth values using spatial and temporal cues
- Applies soft transitions instead of hard occlusion in uncertain regions
  This avoids harsh visual artifacts and improves perceptual quality.

## 6. Performance Considerations

- Target ≥ 30 FPS
- Low latency
- Efficient GPU usage
- Asynchronous pipeline (camera, inference, rendering decoupled)

## 7. Challenges and Solutions

- Noisy depth → filtering
- Flickering → temporal smoothing and confidence aware composting
- Latency → optimization, resolution scaling, GPU-based compositing
- Alignment Between CV and Rendering → Solved by maintaining consistent camera intrinsics and coordinate systems across all stages.

## 8. Results

- Real-time occlusion
- Stable compositing
- Improved realism

## 9. Future Work

- Multi-view reconstruction
- Gaze-based rendering
- Mobile deployment
- Improve robustness in dynamic scenes

## 10. Key Takeaways

- CV + Graphics integration
- Real-time system design
- Perceptual rendering principles
