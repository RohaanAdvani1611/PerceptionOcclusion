"""Task 15 experiment: the two CPU ops in front of the first FP16 baseline. Run after convert.py:
    python experiment_input_ops.py

The first baseline ("scale_outside") had ImageType(scale=1/255), so Core ML scaled the image
in FP32 (mul) and converted it to FP16 (cast), and neither runs on the Neural Engine. The
current baseline folds the 1/255 into the model's normalization instead. Also tried:
  tensor_input  an FP16 tensor input, so the app converts frames itself
Writes both to coreml/models/experiments/ and compares placement, accuracy and Python latency
with the current baseline.
"""
import time

import coremltools as ct
import numpy as np
from PIL import Image

from analyze_placement import plan_ops
from convert import DEPTH_IMAGE_OUTPUT, SHAPE, convert, trace
from midas_loader import MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

EXPERIMENTS = MODELS_DIR / "experiments"
UNITS = ct.ComputeUnit.CPU_AND_NE


def build():
    midas = load_midas_small()
    example = to_wrapper_input(load_frames("calibration")[0][1])
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    print("midas_small_fp16_scale_outside")
    traced = trace(MidasForCoreML(midas, input_range=1.0).eval(), example / 255)
    scaled_input = ct.ImageType(name="image", shape=SHAPE, scale=1 / 255.0, color_layout=ct.colorlayout.BGR)
    convert(traced, ct.precision.FLOAT16, DEPTH_IMAGE_OUTPUT, scaled_input).save(
        str(EXPERIMENTS / "midas_small_fp16_scale_outside.mlpackage"))

    print("midas_small_fp16_tensor_input")
    traced = trace(MidasForCoreML(midas).eval(), example)
    tensor_input = ct.TensorType(name="image", shape=SHAPE, dtype=np.float16)
    convert(traced, ct.precision.FLOAT16, DEPTH_IMAGE_OUTPUT, tensor_input).save(
        str(EXPERIMENTS / "midas_small_fp16_tensor_input.mlpackage"))


def feed(name, rgb):
    if name.endswith("tensor_input"):
        return {"image": np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float16)}
    return {"image": Image.fromarray(rgb)}


def main():
    build()
    paths = {
        "midas_small_fp16": MODELS_DIR / "midas_small_fp16.mlpackage",
        "midas_small_fp16_scale_outside": EXPERIMENTS / "midas_small_fp16_scale_outside.mlpackage",
        "midas_small_fp16_tensor_input": EXPERIMENTS / "midas_small_fp16_tensor_input.mlpackage",
    }
    models = {name: ct.models.MLModel(str(path), compute_units=UNITS) for name, path in paths.items()}

    print(f"\nPlacement with {UNITS.name}:")
    for name, model in models.items():
        rows = plan_ops(model.get_compiled_model_path(), UNITS)
        off = [f"{r['op']} on {r['device']}" for r in rows if r["device"] != "ANE"]
        print(f"  {name:32} {len(rows)} ops, off the ANE: {', '.join(off) or 'none'}")

    frames = load_frames("heldout")
    outputs = {name: [np.asarray(m.predict(feed(name, rgb))["depth"], np.float32) for _, rgb in frames]
               for name, m in models.items()}
    print("\nAgainst the baseline on the held-out frames:")
    for name in list(models)[1:]:
        pearson = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(outputs[name], outputs["midas_small_fp16"])]
        print(f"  {name:32} Pearson mean {np.mean(pearson):.6f}  worst {np.min(pearson):.6f}")

    # Rounds alternate between models, so heat and background load affect them equally
    inputs = {name: [feed(name, rgb) for _, rgb in frames[:20]] for name in models}
    times = {name: [] for name in models}
    for name, model in models.items():
        for i in range(30):
            model.predict(inputs[name][i % 20])
    for _ in range(10):
        for name, model in models.items():
            for i in range(40):
                start = time.perf_counter()
                model.predict(inputs[name][i % 20])
                times[name].append(1000 * (time.perf_counter() - start))
    print("\nPython predict time, rough (Stage 6 measures properly in Swift):")
    for name, t in times.items():
        print(f"  {name:32} p50 {np.median(t):.3f} ms  p90 {np.percentile(t, 90):.3f} ms  (n={len(t)})")


if __name__ == "__main__":
    main()
