"""Converts MiDaS small to Core ML. Run: python convert.py

Writes to coreml/models/:
  midas_small_fp32.mlpackage  reference for numerical checks only, with a tensor output
  midas_small_fp16.mlpackage  the baseline the app uses, with a Float16 grayscale image output
"""
import time

import coremltools as ct
import torch

from midas_loader import INPUT_SIZE, MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
# Camera frames arrive as BGRA, so declaring BGR lets Core ML take them without a
# colour conversion. No scale: setting one adds two CPU ops in front of the model, so the
# wrapper does the 1/255 instead (task 15).
IMAGE_INPUT = ct.ImageType(name="image", shape=SHAPE, color_layout=ct.colorlayout.BGR)
DEPTH_IMAGE_OUTPUT = ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)


def trace(wrapper, example):
    with torch.no_grad():
        # The backbone's "same" padding works out its sizes on the first run and caches them.
        # Running once first means the trace records fixed padding instead of shape arithmetic.
        wrapper(example)
        return torch.jit.trace(wrapper, example)


def convert(traced, precision, output, image_input=IMAGE_INPUT):
    start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[image_input],
        outputs=[output],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = "MiDaS v2.1 small (Intel ISL), converted with coremltools"
    mlmodel.short_description = "Relative inverse depth: larger values are closer. Scale and shift are unknown."
    mlmodel.input_description["image"] = "256x256 BGR frame, stretched from the camera image"
    mlmodel.output_description["depth"] = "256x256 relative inverse depth"
    print(f"  converted in {time.perf_counter() - start:.1f} s")
    return mlmodel


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    wrapper = MidasForCoreML(load_midas_small()).eval()
    traced = trace(wrapper, to_wrapper_input(load_frames("calibration")[0][1]))

    print("FP32 reference")
    fp32 = convert(traced, ct.precision.FLOAT32, ct.TensorType(name="depth"))
    fp32.save(str(MODELS_DIR / "midas_small_fp32.mlpackage"))

    print("FP16 baseline")
    fp16 = convert(traced, ct.precision.FLOAT16, DEPTH_IMAGE_OUTPUT)
    fp16.save(str(MODELS_DIR / "midas_small_fp16.mlpackage"))

    for name in ("midas_small_fp32", "midas_small_fp16"):
        path = MODELS_DIR / f"{name}.mlpackage"
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{name}.mlpackage: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
