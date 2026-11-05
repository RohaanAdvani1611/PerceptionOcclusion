"""Builds the three compressed versions from the FP16 baseline. Run after convert.py:
    python compress.py

Writes to coreml/models/:
  midas_small_w8.mlpackage    weights stored as 8-bit integers, one scale per output channel,
                              except the output head's two 3x3 convs, which stay FP16
  midas_small_p6.mlpackage    weights stored as 6-bit indices into a 64-entry palette per tensor
  midas_small_w8a8.mlpackage  8-bit weights, plus activations rounded to 8 bits using ranges
                              measured on the 150 calibration frames
and to coreml/models/experiments/:
  midas_small_w8_naive.mlpackage  every weight 8-bit. Its output head produces inf and NaN on
                                  the Neural Engine for some frames, though not on CPU or GPU.

Weight tensors with fewer than 2,048 values (coremltools' default threshold) stay FP16.
"""
import time

import coremltools as ct
import coremltools.optimize.coreml as cto
from PIL import Image

from midas_loader import MODELS_DIR, load_frames

BASELINE = MODELS_DIR / "midas_small_fp16.mlpackage"
# The output head carries the network's largest activations (up to about 7,300). With 8-bit
# weights these two convs overflow on the Neural Engine. Keeping them FP16 adds about 28 KB.
OUTPUT_HEAD = ["midas_scratch_output_conv_0_weight_to_fp16", "midas_scratch_output_conv_2_weight_to_fp16"]


def weights_8bit(model, keep_fp16=()):
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"),
        op_name_configs={name: None for name in keep_fp16},
    )
    return cto.linear_quantize_weights(model, config)


def weights_8bit_keep_head(model):
    return weights_8bit(model, keep_fp16=OUTPUT_HEAD)


def palettize_6bit(model):
    config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=6))
    return cto.palettize_weights(model, config)


def w8a8(model):
    # Activations first, while the weights are still FP16, then weights: the order coremltools documents
    frames = load_frames("calibration")
    sample_data = [{"image": Image.fromarray(rgb)} for _, rgb in frames]
    config = cto.OptimizationConfig(global_config=cto.experimental.OpActivationLinearQuantizerConfig(
        mode="linear_symmetric"))
    print(f"  calibrating activations on {len(sample_data)} frames")
    return weights_8bit(cto.experimental.linear_quantize_activations(model, config, sample_data))


def package_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    baseline = ct.models.MLModel(str(BASELINE))
    print(f"midas_small_fp16 (baseline): {package_mb(BASELINE):.1f} MB")
    (MODELS_DIR / "experiments").mkdir(exist_ok=True)
    for name, build in (("midas_small_w8", weights_8bit_keep_head), ("midas_small_p6", palettize_6bit),
                        ("midas_small_w8a8", w8a8), ("experiments/midas_small_w8_naive", weights_8bit)):
        print(name)
        start = time.perf_counter()
        path = MODELS_DIR / f"{name}.mlpackage"
        build(baseline).save(str(path))
        print(f"  built in {time.perf_counter() - start:.1f} s, {package_mb(path):.1f} MB")


if __name__ == "__main__":
    main()
