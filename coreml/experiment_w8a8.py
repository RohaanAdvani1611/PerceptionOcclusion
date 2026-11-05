"""Stage 4 experiment: where W8A8 loses its accuracy. Run after compress.py:
    python experiment_w8a8.py

Plain W8A8 loses far more accuracy than 8-bit weights alone. Three versions isolate why:
  a8_only            activations 8-bit, weights FP16: is it the weights or the activations?
  w8a8_head_fp16     output head activations left FP16 (they reach about 7,300, so 8-bit steps are ~57)
  w8a8_backbone_only only the EfficientNet backbone's activations are 8-bit, the decoder stays FP16
  w8a8_adds_fp16     the inputs of every add (residual and skip connections) stay FP16
All keep the output head's weights FP16, like midas_small_w8. coremltools rounds only the inputs
of conv, add and pooling ops, and excluding an op keeps its inputs FP16. Writes them to
coreml/models/experiments/, skipping any already built, and prints accuracy on the held-out
frames with rough Python latency.
"""
import time

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from PIL import Image

from compress import BASELINE, weights_8bit_keep_head
from evaluate import metrics
from midas_loader import MODELS_DIR, MidasForCoreML, load_frames, load_midas_small, to_wrapper_input

EXPERIMENTS = MODELS_DIR / "experiments"
HEAD_START = "input_451_cast_fp16"                        # output_conv.0, the first output head op
DECODER_START_WEIGHT = "midas_scratch_layer1_rn_weight_to_fp16"  # first decoder conv's weight


def op_names_of_type(model, op_type):
    block = list(model.get_spec().mlProgram.functions["main"].block_specializations.values())[0]
    return [op.attributes["name"].immediateValue.tensor.strings.values[0]
            for op in block.operations if op.type == op_type]


def op_names_from(model, start):
    """Names of the non-const ops from `start` to the end, in program order. `start` is an op name,
    or the name of the weight const used by the first op."""
    block = list(model.get_spec().mlProgram.functions["main"].block_specializations.values())[0]
    names, found = [], False
    for op in block.operations:
        if op.type == "const":
            continue
        name = op.attributes["name"].immediateValue.tensor.strings.values[0]
        weight = op.inputs["weight"].arguments[0].name if "weight" in op.inputs else None
        found = found or start in (name, weight)
        if found:
            names.append(name)
    assert names, f"{start} not found"
    return names


def quantize_activations(model, sample_data, keep_fp16=()):
    config = cto.OptimizationConfig(
        global_config=cto.experimental.OpActivationLinearQuantizerConfig(mode="linear_symmetric"),
        op_name_configs={name: None for name in keep_fp16},
    )
    return cto.experimental.linear_quantize_activations(model, config, sample_data)


def main():
    baseline = ct.models.MLModel(str(BASELINE))
    sample_data = [{"image": Image.fromarray(rgb)} for _, rgb in load_frames("calibration")]
    head = op_names_from(baseline, HEAD_START)
    decoder = op_names_from(baseline, DECODER_START_WEIGHT)
    adds = op_names_of_type(baseline, "add")
    print(f"output head: {len(head)} ops, decoder: {len(decoder)} ops, adds: {len(adds)}")

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    builds = {
        "a8_only": lambda: quantize_activations(baseline, sample_data, keep_fp16=head),
        "w8a8_head_fp16": lambda: weights_8bit_keep_head(quantize_activations(baseline, sample_data, keep_fp16=head)),
        "w8a8_backbone_only": lambda: weights_8bit_keep_head(quantize_activations(baseline, sample_data, keep_fp16=decoder)),
        "w8a8_adds_fp16": lambda: weights_8bit_keep_head(quantize_activations(baseline, sample_data, keep_fp16=adds)),
    }
    for name, build in builds.items():
        path = EXPERIMENTS / f"midas_small_{name}.mlpackage"
        if path.exists():
            continue
        start = time.perf_counter()
        build().save(str(path))
        print(f"  built {name} in {time.perf_counter() - start:.0f} s")

    frames = load_frames("heldout")
    wrapper = MidasForCoreML(load_midas_small()).eval()
    with torch.no_grad():
        reference = [wrapper(to_wrapper_input(rgb)).numpy().reshape(256, 256) for _, rgb in frames]
    paths = {"w8": MODELS_DIR / "midas_small_w8.mlpackage", "w8a8": MODELS_DIR / "midas_small_w8a8.mlpackage"}
    paths.update({name: EXPERIMENTS / f"midas_small_{name}.mlpackage" for name in builds})

    print(f"\n{len(frames)} held-out frames, CPU_AND_NE, against PyTorch FP32")
    print(f"{'model':20} {'broken':>6} {'AbsRel mean':>12} {'worst':>7} {'mask flip mean':>15} {'worst':>7} {'p50 ms':>7}")
    for name, path in paths.items():
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        inputs = [{"image": Image.fromarray(rgb)} for _, rgb in frames]
        preds = [np.asarray(model.predict(x)["depth"], np.float32) for x in inputs]
        times = []
        for x in inputs * 3:
            start = time.perf_counter()
            model.predict(x)
            times.append(1000 * (time.perf_counter() - start))
        good = [i for i, p in enumerate(preds) if np.isfinite(p).all()]
        per_frame = [metrics(preds[i], reference[i]) for i in good]
        absrel = np.array([m["absrel"] for m in per_frame])
        mask = np.array([m["mask"] for m in per_frame])
        print(f"{name:20} {len(preds) - len(good):6} {100 * absrel.mean():11.2f}% {100 * absrel.max():6.1f}% "
              f"{100 * mask.mean():14.2f}% {100 * mask.max():6.1f}% {np.median(times):7.2f}")


if __name__ == "__main__":
    main()
