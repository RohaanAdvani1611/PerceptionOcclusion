"""Measures each model's accuracy on the held-out frames against PyTorch FP32. Run after compress.py:
    python evaluate.py

Metrics, per frame:
  AbsRel    mean |aligned - ref| / ref, after a least-squares scale and shift aligns the output to
            the reference (MiDaS has no fixed scale). Pixels below the reference's 1st percentile
            are skipped, since dividing by near-zero values blows the number up.
  Pearson   correlation with the reference.
  Mask      share of pixels where the occlusion decision differs at threshold 0.5, after the same
            1st-99th percentile normalization DepthEstimator.cpp applies.

Core ML models run with CPU_AND_NE, the setting the app uses. The C++ pipeline's ONNX model is
included for comparison, and so is the naive W8 experiment. Frames with any inf or NaN are
counted as broken and left out of the other metrics. Writes coreml/results/accuracy.csv and, for
each model, an image of its worst frame to coreml/results/comparisons/.
"""
import csv

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from check_conversion import ONNX_PATH, colorize
from midas_loader import COREML_DIR, MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

VARIANTS = {"fp16": "midas_small_fp16", "w8": "midas_small_w8", "p6": "midas_small_p6",
            "w8a8": "midas_small_w8a8", "w8_naive": "experiments/midas_small_w8_naive"}
THRESHOLD = 0.5
RESULTS_DIR = COREML_DIR / "results"


def align(pred, ref, valid):
    """Least-squares scale and shift so pred best matches ref on the valid pixels."""
    a = np.stack([pred[valid], np.ones(valid.sum())], axis=1)
    scale, shift = np.linalg.lstsq(a, ref[valid], rcond=None)[0]
    return scale * pred + shift


def occlusion_mask(depth):
    """Real objects hide the sphere where normalized depth is above the threshold, as in the C++ app."""
    low, high = np.percentile(depth, [1, 99])
    return np.clip((depth - low) / (high - low), 0, 1) > THRESHOLD


def metrics(pred, ref):
    valid = ref > np.percentile(ref, 1)
    aligned = align(pred, ref, valid)
    return {
        "absrel": np.mean(np.abs(aligned[valid] - ref[valid]) / ref[valid]),
        "pearson": np.corrcoef(pred.ravel(), ref.ravel())[0, 1],
        "mask": np.mean(occlusion_mask(pred) != occlusion_mask(ref)),
    }


def worst_frame_image(name, rgb, ref, pred, label):
    valid = ref > np.percentile(ref, 1)
    error = np.abs(align(pred, ref, valid) - ref) / ref
    error_image = cv2.applyColorMap((255 * np.clip(error / 0.10, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # Grey where both agree the real scene is in front, red and blue where they disagree
    ref_mask, pred_mask = occlusion_mask(ref), occlusion_mask(pred)
    mask_image = np.zeros((*ref.shape, 3), np.uint8)
    mask_image[ref_mask & pred_mask] = (128, 128, 128)
    mask_image[pred_mask & ~ref_mask] = (0, 0, 255)
    mask_image[ref_mask & ~pred_mask] = (255, 128, 0)
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(ref), colorize(pred), error_image, mask_image]
    labels = [name, "PyTorch FP32", label, "error, 10% = max", "mask: red/blue = flipped"]
    for tile, text in zip(tiles, labels):
        cv2.putText(tile, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)


def main():
    frames = load_frames("heldout")
    wrapper = MidasForCoreML(load_midas_small()).eval()
    with torch.no_grad():
        reference = [wrapper(to_wrapper_input(rgb)).numpy().reshape(256, 256) for _, rgb in frames]

    outputs = {}
    for variant, path in VARIANTS.items():
        model = ct.models.MLModel(str(MODELS_DIR / f"{path}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        outputs[variant] = [np.asarray(model.predict({"image": Image.fromarray(rgb)})["depth"], np.float32)
                            for _, rgb in frames]
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs["onnx"] = [session.run(None, {input_name: (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0]
                       .reshape(256, 256) for _, rgb in frames]

    comparisons = RESULTS_DIR / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{len(frames)} held-out frames, against PyTorch FP32\n")
    print(f"{'model':8} {'broken':>6} {'AbsRel mean':>12} {'worst':>8} {'Pearson mean':>13} {'worst':>9} "
          f"{'mask flip mean':>15} {'worst':>7}  worst frame")
    for name, preds in outputs.items():
        good = [i for i, p in enumerate(preds) if np.isfinite(p).all()]
        per_frame = [metrics(preds[i], reference[i]) for i in good]
        absrel = np.array([m["absrel"] for m in per_frame])
        pearson = np.array([m["pearson"] for m in per_frame])
        mask = np.array([m["mask"] for m in per_frame])
        worst = good[int(absrel.argmax())]
        row = {"model": name, "broken_frames": len(preds) - len(good),
               "absrel_mean": absrel.mean(), "absrel_worst": absrel.max(),
               "pearson_mean": pearson.mean(), "pearson_worst": pearson.min(),
               "mask_flip_mean": mask.mean(), "mask_flip_worst": mask.max(), "worst_frame": frames[worst][0]}
        rows.append(row)
        print(f"{name:8} {row['broken_frames']:6} {100 * row['absrel_mean']:11.3f}% {100 * row['absrel_worst']:7.3f}% "
              f"{row['pearson_mean']:13.6f} {row['pearson_worst']:9.6f} "
              f"{100 * row['mask_flip_mean']:14.3f}% {100 * row['mask_flip_worst']:6.2f}%  {row['worst_frame']}")
        label = "ONNX (C++ pipeline)" if name == "onnx" else f"Core ML {name.upper()} (CPU+NE)"
        image = worst_frame_image(frames[worst][0], frames[worst][1], reference[worst], preds[worst], label)
        cv2.imwrite(str(comparisons / f"worst_{name}.png"), image)

    with open(RESULTS_DIR / "accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\nWrote {RESULTS_DIR / 'accuracy.csv'} and worst-frame images in {comparisons}")


if __name__ == "__main__":
    main()
