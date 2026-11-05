# MiDaS on Core ML: task list

**Goal:** get the MiDaS depth model running well on Apple hardware through Core ML, ideally on the Neural Engine. Build four versions (FP16, 8-bit weights, 6-bit palettized and W8A8), and measure each one's size, speed, accuracy and processor placement on the M2 Mac and iPhone 14. Run them on live camera frames in a small Swift app.

The end result is `RESULTS.md`, with an explanation of why each version performed as it did and five interview talking points with real numbers. The number to beat is about 40 ms per frame for the model on CPU with ONNX.

See [PLAN.md](PLAN.md) for details. Tasks marked **(You)** need you at the keyboard or in Xcode. Each stage ends with a check before the next one starts.

## Done

- [x] Understand the codebase (Phase 0)
- [x] Fix the 4 pipeline issues and re-benchmark
- [x] Check the machine (Phase 1)
- [x] Write `PLAN.md` (Phase 2)

## Stage 1: Prepare (Phase 3) (done)

- [x] Share the iPhone's iOS version (27.0.1)
- [x] 1. **(You)** Approve the plan
- [x] 1a. **(You)** Test that Xcode 26.3 can install a blank app on the iPhone on iOS 27 (it can)
- [x] 2. Add the new exclusions to `.gitignore`
- [x] 3. Create the Python 3.12 environment with pinned PyTorch 2.7.0 and coremltools 9.0
- [x] 4. Download the MiDaS source and official weights, and record their exact versions
- [x] 5. Write `capture_frames.py`
- [x] 6. **(You)** Capture 150 calibration frames and 100 held-out frames, from different scenes

## Stage 2: Convert (Phase 3) (done)

- [x] 7. Wrap the model so normalization happens exactly once, with image input and Float16 image output
- [x] 8. Convert it to an FP32 reference model and an FP16 baseline
- [x] 9. Check Core ML against PyTorch on the same frames (if this fails, stop and debug)
- [x] 10. Compare PyTorch with the corrected ONNX model to measure the resize gap
- [x] 11. **(You)** Open the `.mlpackage` in Xcode and confirm its inputs and outputs

## Stage 3: Neural Engine analysis (Phase 4) (done)

- [x] 12. Generate the per-op placement table, by op count and by estimated cost
- [x] 13. **(You)** Run Xcode's performance report on the baseline
- [x] 14. Check FP16 for overflow and `inf` or `NaN`, including on the worst frame
- [x] 15. Explain every op that isn't on the Neural Engine, and run a separate experiment for any fallback worth fixing

## Stage 4: Compress and evaluate (Phase 5) (done)

- [x] 16. Build the 8-bit, 6-bit palettized and W8A8 versions
- [x] 17. Measure each version's accuracy on the held-out frames, including occlusion mask disagreement
- [x] 18. Generate side-by-side images, including each version's worst frame
- [x] 19. **(You)** Inspect the images and accuracy table

## Stage 5: Swift app on Mac (Phase 6)

- [ ] 20. **(You)** Create the `DepthCam` Xcode project using the click-by-click steps
- [ ] 21. Write the camera capture with frame dropping, plus the vImage resize
- [ ] 22. Write the model loading and the Metal depth display
- [ ] 23. Add the FPS and latency labels, plus the model and compute-unit pickers
- [ ] 24. Add the benchmark mode and `MLComputePlan` report
- [ ] 25. **(You)** Run the app, allow camera access, and confirm live depth and model switching work

## Stage 6: Mac benchmarks (Phase 7)

- [ ] 26. Write the C++ ONNX timing runner, so the comparison uses the same frames
- [ ] 27. **(You)** Run all 16 combinations of version and compute unit, plugged in with other apps closed
- [ ] 28. **(You)** Run Xcode performance reports for the four versions
- [ ] 29. **(You)** Measure power with `powermetrics`, and run the 10-minute soak test on the best version

## Stage 7: iPhone 14 (Phase 8)

- [ ] 30. **(You)** Connect the iPhone, sign in with an Apple ID, turn on Developer Mode and install the app
- [ ] 31. **(You)** Confirm live depth works on the phone
- [ ] 32. **(You)** Repeat the 16 benchmarks, the performance reports, the power check and the soak test
- [ ] 33. Compare the Mac and iPhone results

## Stage 8: Write up (Phase 9)

- [ ] 34. Write `RESULTS.md`, with the tables, explanations and ONNX comparison
- [ ] 35. Write the five talking points using only measured numbers
- [ ] 36. **(Optional)** Add the temporal filter or occlusion demo to the app
