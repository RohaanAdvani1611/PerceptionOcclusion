"""Checks the converted models against PyTorch on every captured frame. Run after convert.py:
    python check_conversion.py

Every model gets the same 256x256 frames, so differences come from the models, not resizing.
Writes coreml/results/conversion_check.png, which shows one of your frames.
"""
import sys
import time

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from midas_loader import COREML_DIR, MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

ONNX_PATH = COREML_DIR.parent / "assets" / "models" / "midas.onnx"
RESULTS_DIR = COREML_DIR / "results"
SAMPLE_FRAME = "bedroom_0025"

# Starting targets from PLAN.md, not published limits
FP32_MIN_PEARSON = 0.9999
FP16_MIN_PEARSON = 0.999


def compare(name, outputs, reference):
    """Prints agreement with the reference and returns per-frame Pearson (-1 if any non-finite)."""
    pearson = np.array([np.corrcoef(o.ravel(), r.ravel())[0, 1] for o, r in zip(outputs, reference)])
    # Max difference as a share of each reference frame's range, so frames are comparable
    rel = np.array([np.abs(o - r).max() / (r.max() - r.min()) for o, r in zip(outputs, reference)])
    finite = all(np.isfinite(o).all() for o in outputs)
    print(f"  {name:34} Pearson mean {pearson.mean():.6f}  worst {pearson.min():.6f}"
          f"  max diff {100 * rel.max():5.2f}% of range  finite: {finite}")
    return pearson if finite else np.full_like(pearson, -1.0)


def run_coreml(variant, units, frames):
    model = ct.models.MLModel(str(MODELS_DIR / f"midas_small_{variant}.mlpackage"), compute_units=units)
    outputs, times = [], []
    for _, rgb in frames:
        image = Image.fromarray(rgb)
        start = time.perf_counter()
        depth = model.predict({"image": image})["depth"]
        times.append(time.perf_counter() - start)
        outputs.append(np.asarray(depth, dtype=np.float32).reshape(256, 256))
    return outputs, 1000 * np.median(times)


def colorize(depth):
    scaled = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def main():
    frames = load_frames("calibration") + load_frames("heldout")
    print(f"{len(frames)} frames\n")

    wrapper = MidasForCoreML(load_midas_small()).eval()
    with torch.no_grad():
        reference = [wrapper(to_wrapper_input(rgb)).numpy().reshape(256, 256) for _, rgb in frames]

    results = {}
    runs = [("fp32", ct.ComputeUnit.CPU_ONLY), ("fp16", ct.ComputeUnit.CPU_ONLY),
            ("fp16", ct.ComputeUnit.CPU_AND_GPU), ("fp16", ct.ComputeUnit.CPU_AND_NE)]
    for variant, units in runs:
        results[variant, units.name], ms = run_coreml(variant, units, frames)
        print(f"  ran {variant} {units.name:12} median {ms:5.1f} ms per frame (Python, rough)")

    print("\nAgainst PyTorch FP32:")
    conversion_ok = compare("Core ML FP32, CPU", results["fp32", "CPU_ONLY"], reference).min() > FP32_MIN_PEARSON
    for units in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
        compare(f"Core ML FP16, {units}", results["fp16", units], reference)

    # Below-target FP16 is a finding to explain (task 14), not a broken conversion
    print("\nAgainst Core ML FP32 (precision loss only):")
    fp32 = results["fp32", "CPU_ONLY"]
    below_target = []
    for units in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
        pearson = compare(f"FP16, {units}", results["fp16", units], fp32)
        if (pearson < FP16_MIN_PEARSON).any():
            below_target.append(f"{units} ({(pearson < FP16_MIN_PEARSON).sum()} frames)")

    # The C++ pipeline's model, given the same frames as RGB in [0, 1]; it normalizes itself
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx = [session.run(None, {input_name: (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0]
            .reshape(256, 256) for _, rgb in frames]
    print("\nC++ pipeline's ONNX model against PyTorch FP32 (the resize gap):")
    onnx_pearson = compare("ONNX", onnx, reference)

    # One typical frame, then the frame where ONNX disagrees most with PyTorch
    names = [name for name, _ in frames]
    rows = []
    for i in (names.index(SAMPLE_FRAME), int(onnx_pearson.argmin())):
        ane = results["fp16", "CPU_AND_NE"][i]
        diff = np.abs(ane - reference[i]) / (0.01 * np.ptp(reference[i]))
        tiles = [cv2.cvtColor(frames[i][1], cv2.COLOR_RGB2BGR), colorize(reference[i]), colorize(ane),
                 colorize(onnx[i]), cv2.applyColorMap((255 * np.clip(diff, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)]
        labels = [names[i], "PyTorch FP32", "Core ML FP16 (CPU+NE)", f"ONNX, Pearson {onnx_pearson[i]:.3f}",
                  "|Core ML - PyTorch|, 1% = max"]
        for tile, label in zip(tiles, labels):
            cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    RESULTS_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(RESULTS_DIR / "conversion_check.png"), np.vstack(rows))
    print(f"\nWrote {RESULTS_DIR / 'conversion_check.png'} ({SAMPLE_FRAME} and {names[int(onnx_pearson.argmin())]})")

    if not conversion_ok:
        print("\nCONVERSION CHECK FAILED: Core ML FP32 does not match PyTorch")
        sys.exit(1)
    if below_target:
        print(f"\nWARNING: FP16 below Pearson {FP16_MIN_PEARSON} on {', '.join(below_target)}")
    print("\nCONVERSION CHECK PASSED")


if __name__ == "__main__":
    main()
