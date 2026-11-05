"""Stage 4 experiment: 6-bit palettization with one palette per group of channels. Run after compress.py:
    python experiment_p6.py

midas_small_p6 uses coremltools' default, one 64-entry palette per weight tensor. Giving each
group of 16 output channels its own palette should fit the weights better for a few KB more.
Grouped palettes need the iOS 18 / macOS 15 model format, so this converts the baseline again
with that target. Writes coreml/models/experiments/midas_small_p6_grouped.mlpackage.
"""
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from PIL import Image

from compress import package_mb
from convert import DEPTH_IMAGE_OUTPUT, convert, trace
from evaluate import metrics
from midas_loader import MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

GROUPED = MODELS_DIR / "experiments" / "midas_small_p6_grouped.mlpackage"


def main():
    wrapper = MidasForCoreML(load_midas_small()).eval()
    traced = trace(wrapper, to_wrapper_input(load_frames("calibration")[0][1]))
    fp16_ios18 = convert(traced, ct.precision.FLOAT16, DEPTH_IMAGE_OUTPUT, target=ct.target.iOS18)
    config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        mode="kmeans", nbits=6, granularity="per_grouped_channel", group_size=16))
    cto.palettize_weights(fp16_ios18, config).save(str(GROUPED))

    frames = load_frames("heldout")
    with torch.no_grad():
        reference = [wrapper(to_wrapper_input(rgb)).numpy().reshape(256, 256) for _, rgb in frames]
    print(f"\n{len(frames)} held-out frames, CPU_AND_NE, against PyTorch FP32")
    print(f"{'model':16} {'size':>8} {'broken':>6} {'AbsRel mean':>12} {'worst':>7} {'mask flip mean':>15} {'worst':>7}")
    for name, path in (("p6", MODELS_DIR / "midas_small_p6.mlpackage"), ("p6_grouped", GROUPED)):
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        preds = [np.asarray(model.predict({"image": Image.fromarray(rgb)})["depth"], np.float32) for _, rgb in frames]
        good = [i for i, p in enumerate(preds) if np.isfinite(p).all()]
        per_frame = [metrics(preds[i], reference[i]) for i in good]
        absrel = np.array([m["absrel"] for m in per_frame])
        mask = np.array([m["mask"] for m in per_frame])
        print(f"{name:16} {package_mb(path):6.1f}MB {len(preds) - len(good):6} {100 * absrel.mean():11.2f}% "
              f"{100 * absrel.max():6.1f}% {100 * mask.mean():14.2f}% {100 * mask.max():6.1f}%")


if __name__ == "__main__":
    main()
