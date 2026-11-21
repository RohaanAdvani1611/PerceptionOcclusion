# MiDaS on Core ML: results

I took the MiDaS v2.1 small depth model and converted it from PyTorch to Core ML, Apple's on-device ML framework. I made four versions of it and ran each one on an M2 MacBook Air and an iPhone 14, in a small Swift camera app called DepthCam. The number to beat was the existing C++ pipeline, which runs the same model with ONNX Runtime on the CPU in about **40 ms per frame**.

**Short answer:** on Apple's Neural Engine the model runs in **1.4–1.8 ms on the Mac** and **2.2–2.8 ms on the iPhone**. That's 14–28× faster than ONNX, and the whole live app uses **26× less power** than the ONNX model alone. Compression makes the model smaller and a bit faster, but the more aggressive versions cost visible accuracy. Plugged back into the C++ occlusion engine, the W8 model **doubled the engine's frame rate** (about 11.5 → 23 FPS).

All raw numbers are in `inference/coreml/` and `coreml/results/accuracy.csv`. The details and every side experiment are in [PLAN.md](PLAN.md).

## Words used below

- **Core ML model (`.mlpackage`):** the converted model file. Xcode compiles it into the app.
- **Compute units:** which processors Core ML is allowed to use:
  - **CPU only**
  - **CPU + GPU**
  - **CPU + Neural Engine**
  - **All**
  Core ML then decides where each operation actually runs.
- **ANE (Apple Neural Engine):** a processor built only for neural networks. It's fast and low-power, but it supports fewer operations and data types than the CPU or GPU.
- **Placement:** which processor each operation of the model runs on. `MLComputePlan` (in code) and Xcode's performance report (in the app) both show it.
- **FP16 / 8-bit / 6-bit:** how many bits store each weight. Fewer bits make a smaller file, but the weights are less precise.
- **p50 / p99:** the median time, and the time that 99% of runs beat.
- **Mask flips:** the share of pixels whose "near or far" decision changes at a 0.5 threshold, compared with the original PyTorch model. This is the error that matters for the occlusion app, since a flipped pixel draws a virtual object in front of or behind the wrong thing.

## The four versions

| Version | What it is | Size |
|---|---|---|
| **FP16** | Every weight in 16-bit floating point. This is the baseline. | 33.3 MB |
| **W8** | Weights stored as 8-bit integers, except two layers kept in FP16 (see Why below) | 16.9 MB |
| **P6** | Weights stored as 6-bit indices into a table of 64 values ("palettized") | 12.7 MB |
| **W8A8** | 8-bit weights, plus 8-bit activations (the values passed between layers). An experiment. | 17.0 MB |

## Results

### Accuracy (100 held-out frames, compared with the original PyTorch model)

| Version | Mask flips, mean | Mask flips, worst frame |
|---|---|---|
| FP16 | 0.17% | 3.1% |
| W8 | 1.05% | 11.9% |
| P6 | 4.58% | 22.9% |
| W8A8 | 8.91% | 86.8% |
| ONNX (old C++ pipeline) | 6.73% | 43.3% |

### Speed (p50, 500 runs back to back on real frames)

| Version | Mac: Neural Engine | Mac: GPU | Mac: CPU | iPhone: Neural Engine | iPhone: GPU | iPhone: CPU |
|---|---|---|---|---|---|---|
| FP16 | 1.81 ms | 8.05 ms | 11.08 ms | 2.79 ms | 19.14 ms | 12.55 ms |
| W8 | 1.69 ms | 8.47 ms | 11.13 ms | 2.49 ms | 19.03 ms | 12.70 ms |
| P6 | 1.59 ms | 8.43 ms | 11.10 ms | 2.20 ms | 19.02 ms | 12.68 ms |
| W8A8 | 1.43 ms | 8.48 ms | 17.31 ms | 2.21 ms | 18.67 ms | 20.91 ms |
| ONNX Runtime, CPU, 4 threads | | | 39.69 ms | | | |

Every Core ML version runs **100% on the Neural Engine** (202 of 202 operations) when the Neural Engine is allowed, on both devices.

### Power, on the Mac (whole app, about 24 frames a second)

| Setting | Power |
|---|---|
| DepthCam, FP16 on the Neural Engine | **0.69 W** |
| DepthCam, W8A8 on the Neural Engine | 0.73 W |
| DepthCam, FP16 on the GPU | 1.54 W |
| DepthCam, FP16 on the CPU only | 1.94 W |
| ONNX model alone, CPU | **18.3 W** |
| Mac idle | 0.25 W |

The iPhone's power tool (Instruments' Power Profiler) only gives relative scores, but it ranks the settings in the same order: Neural Engine lowest, then GPU, then CPU.

### Loading and heat

- **First load on the Neural Engine is slow:** 0.7–2.5 s on the Mac and up to 1.6 s on the iPhone, because Core ML compiles the model for the Neural Engine. It caches the result, so later loads take 20–50 ms.
- **Heat:** in 10-minute live runs, both devices stayed at "nominal" thermal state. The iPhone held a flat 5.1 ms per frame (W8A8). The Mac drifted from about 5.7 to 6.3 ms (W8).

## Why it turned out this way

**1. Why the whole model fits on the Neural Engine.** MiDaS small uses only plain operations the Neural Engine supports: convolutions, activations, additions, padding and resizes. The first conversion still had **2 operations on the CPU**. I had asked Core ML to divide the image by 255 as it came in, and Core ML did that in FP32 on the CPU before handing it to the Neural Engine. Moving the divide inside the model fixed it, bringing placement to 202 of 202 on the Neural Engine with the same accuracy. So a model's input settings can affect placement, not only its layers.

**2. Why smaller versions are faster on the Neural Engine, but not on the GPU or CPU.** On the Neural Engine, each step down in size was faster:

| Version | Mac, Neural Engine |
|---|---|
| FP16 | 1.81 ms |
| W8 | 1.69 ms |
| P6 | 1.59 ms |
| W8A8 | 1.43 ms |

That suggests the Neural Engine spends much of its time moving data, not doing maths, so less data means less time. This is my reading of the timings, not a direct measurement. On the GPU and CPU, the compressed weights are expanded back to 16-bit before use, so there's no gain. W8A8 is actually **56% slower on the Mac's CPU** (17.3 against 11.1 ms), because it adds 387 small "convert to 8-bit and back" operations, and the CPU runs every one of them.

**3. Why W8 needed two layers kept in FP16.** With *every* weight in 8-bit, 48 of 100 frames came out as `inf` or `NaN` on the Neural Engine, while the same model was fine on the CPU and GPU. I traced the failure to the two convolutions in the output head, which carry the network's largest values. Keeping just those two layers in FP16 fixed every frame and cost about 28 KB. **Lesson:** a compressed model can be correct on one processor and broken on another, so test on the processor you'll ship on.

**4. Why accuracy drops as compression increases.** Fewer bits mean each weight is rounded more:
- **W8** has 256 levels per weight, and loses little.
- **P6** has only 64 values for a whole layer, and loses noticeably more.
- **W8A8** also rounds every value passed between layers. Its worst frame flips 87% of the mask, which is unusable. The loss comes from the backbone's activations, but I didn't find the exact cause.

Each version's accuracy is almost the same on the Neural Engine, GPU and CPU, so **the loss comes from the compression, not the processor.**

**5. Why the iPhone is slower.** Both chips have a 16-core Neural Engine with a similar rated speed, but the iPhone's Neural Engine is 1.4–1.5× slower here. Compression also helps the iPhone more: FP16 → P6 saves 21% there, against 12% on the Mac. Both fit the iPhone having less memory bandwidth (LPDDR4X against the M2's LPDDR5), though I didn't measure bandwidth directly. The iPhone 14's 5-core GPU is 2.3× slower than the Mac's 10-core GPU, which is why on the iPhone the GPU is even slower than the CPU.

**6. Why the live app is slower than the benchmark.** The benchmark runs predictions back to back. The live app waits about 40 ms between camera frames, and every processor partly powers down in that gap and has to wake up again. For W8 on the Mac's Neural Engine, a 33 ms pause between runs raised the time from 1.59 to 5.14 ms. On the GPU it went from 6.92 to 14.12 ms, and on the CPU from 10.97 to 16.59 ms. The Neural Engine is still the fastest at camera speed. Benchmark numbers are best cases, so judge a live app at its real frame rate.

**7. Why the ONNX pipeline was so much slower and less accurate.**
- **Speed:** the ONNX model runs in FP32 on the CPU, which is 22× slower than Core ML FP16 on the Neural Engine. Even on the CPU alone, Core ML FP16 is 3.6× faster than ONNX Runtime.
- **Accuracy:** its 6.7% mask flips aren't caused by ONNX itself. The C++ pipeline resizes the camera image differently from the PyTorch reference, and that difference shows up as depth error, worst on blank walls.

## Back in the occlusion engine

The C++ engine now runs the W8 model through Core ML by default (`src/perception/CoreMLDepthEstimator.mm`), and `--depth onnx` brings back the old path. Two 1,200-frame runs were recorded back to back on the M2 with the live camera:

| | ONNX, CPU | Core ML W8, Neural Engine |
|---|---|---|
| Depth step (`infer`, mean) | 44.6 ms | **5.5 ms** |
| Processing latency, mean (p99) | 64.6 ms (83.3) | **27.0 ms (31.0)** |
| Frame rate (from the sum of stage times) | ~11.5 FPS | **~23 FPS** |

- **Why not 8× the frame rate?** The engine runs one stage after another. Once depth stopped being the bottleneck, waiting for the camera (16 ms) and drawing the windows (14 ms) became 70% of each frame. Removing a bottleneck shows you the next one.
- **The depth step's 5.5 ms** includes resizing and scaling the 640×480 frame. On saved frames back to back it takes 3.7 ms. The gap is the same "powers down between frames" effect as above.
- **The depth looks the same or better.** Core ML and ONNX disagree on 6.8% of mask pixels, about the same as ONNX's known 6.7% gap from the original model.

## Things that surprised me

- **Xcode's performance report disagreed with the app on the iPhone.** Xcode put the four input operations on the GPU. For W8A8, it put 530 of 580 operations on the CPU, at 143 ms per prediction. Inside DepthCam on the same phone, W8A8 runs 100% on the Neural Engine at 2.2 ms. My best guess (unverified) is that Xcode's report compiles the model on the phone with iOS 27.2.1's compiler, while the app ships a copy compiled earlier by Xcode 26.3. Either way: **check placement in the shipped app, on the real device.**
- **The Neural Engine drew 0.26–0.35 W in every DepthCam reading on the Mac,** even with the model set to the GPU or CPU. It drew nothing during the ONNX reading, when the camera was off. Something tied to the camera seems to use it, but I didn't find out what.
- **macOS App Nap** throttled DepthCam when it was in the background, which ruined my first power readings. The app now tells macOS it's doing user-initiated work.

## What I'd use

- **For the app: W8.** It's half the size of FP16 and a little faster, with about 1% mask flips.
- **If accuracy matters most: FP16.**
- **P6** is the smallest, but its 4.6% mask flips are visible.
- **W8A8** isn't usable on this hardware. Apple's docs say 8-bit activation maths on the Neural Engine starts with the A17 Pro and M4, so on these chips it can only save on data movement, and the accuracy cost is too high.

## Five interview talking points

1. **"I found depth was 69% of my engine's latency, moved it to the Neural Engine, and doubled the frame rate."**
   - The model runs 202 of 202 operations on the ANE, in 1.81 ms on an M2 against 39.69 ms with ONNX. On an iPhone 14 it takes 2.79 ms.
   - In the live engine, the depth step fell from 44.6 to 5.5 ms, latency from 65 to 27 ms, and the frame rate rose from about 11.5 to 23 FPS.
   - The camera and display are now the bottleneck.

2. **"Input preprocessing can knock ops off the Neural Engine."** Letting Core ML apply a 1/255 scale to the image input put 2 ops on the CPU. Folding the scale into the model's normalization brought it to 100% ANE with unchanged accuracy.

3. **"Compression trades size for accuracy, and the trade is very uneven."**
   - 8-bit weights halved the size (33.3 → 16.9 MB) for about 1% mask flips.
   - 6-bit palettization reached 12.7 MB but 4.6% flips.
   - W8A8 was the fastest on the ANE (1.43 ms) but flipped up to 87% of the mask on one frame.

   I'd ship W8.

4. **"A quantized model can be fine on CPU and GPU but broken on the ANE."** Full 8-bit weights produced `inf` or `NaN` on 48 of 100 frames, only on the ANE. I traced it to the two output-head convolutions with the largest activations, and keeping those in FP16 fixed it for 28 KB. I also saw Xcode's iPhone report put W8A8 mostly on the CPU at 143 ms, while the shipped app ran it fully on the ANE at 2.2 ms. So I always verify placement on the target device.

5. **"Benchmark latency isn't live latency, and the ANE wins on power."**
   - **Latency:** back to back, W8 takes 1.59 ms on the M2's ANE, but at a 30 fps pace it takes 5.14 ms, because processors power down between frames.
   - **Power:** even so, the whole live app on the ANE drew 0.69 W, against 1.54 W on the GPU, 1.94 W on the CPU, and 18.3 W for the ONNX model alone.
   - **Heat:** both devices stayed thermally nominal through 10-minute runs.

## Setup

- **Software:** coremltools 9.0, PyTorch 2.7.0, ML Program format, iOS 17 deployment target, Xcode 26.3.
- **Devices:** M2 MacBook Air (8 GB, macOS 15.6.1), and iPhone 14 (iOS 27.2.1).
- **Data:** 150 captured frames for calibration, and 100 held-out frames for accuracy and timing.
