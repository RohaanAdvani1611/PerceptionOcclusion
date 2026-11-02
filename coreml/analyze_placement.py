"""Reports where Core ML plans to run each op, using MLComputePlan. Run:
    python analyze_placement.py                   # the FP16 baseline
    python analyze_placement.py midas_small_w8    # any model in coreml/models/

For each compute-unit setting it prints the share of ops and of estimated cost on each
device, and lists every op not on the Neural Engine. With CPU_AND_NE it also writes
coreml/results/placement_<model>.csv, one row per op.

This is Core ML's plan, a prediction. Measured latency (Stage 6) checks it.
"""
import csv
import sys
from collections import Counter, defaultdict

import coremltools as ct
from coremltools.models import compute_plan as cp

from midas_loader import COREML_DIR, MODELS_DIR

DEVICE_NAMES = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}
UNITS = [ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_ONLY]


def device_name(device):
    return DEVICE_NAMES[type(device).__name__]


def plan_ops(compiled_path, units):
    """Returns one dict per op that has a device, in program order. Consts have none."""
    plan = cp.MLComputePlan.load_from_path(compiled_path, compute_units=units)
    rows = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        rows.append({
            "op": op.operator_name.split(".")[-1],  # "ios17.conv" -> "conv"
            "output": op.outputs[0].name if op.outputs else "",
            "device": device_name(usage.preferred_compute_device),
            "supported": "/".join(sorted(device_name(d) for d in usage.supported_compute_devices)),
            "cost": cost.weight if cost else 0.0,
        })
    return rows


def summarize(units, rows):
    total_cost = sum(r["cost"] for r in rows)
    count = Counter(r["device"] for r in rows)
    cost = defaultdict(float)
    for r in rows:
        cost[r["device"]] += r["cost"]
    # Each change of device between consecutive ops means handing data between processors
    switches = sum(a["device"] != b["device"] for a, b in zip(rows, rows[1:]))
    shares = "  ".join(f"{d} {100 * count[d] / len(rows):5.1f}% ops, {100 * cost[d] / total_cost:5.1f}% cost"
                       for d in ("ANE", "GPU", "CPU") if count[d])
    print(f"{units.name:12} {len(rows)} ops  {shares}  device switches: {switches}")


def by_op_type(rows):
    """Per op type: count, devices it was placed on, devices it supports, cost share."""
    total_cost = sum(r["cost"] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["op"]].append(r)
    print(f"\n  {'op':24} {'count':>5}  {'placed on':18} {'can run on':18} {'cost %':>6}")
    for op, group in sorted(groups.items(), key=lambda g: -sum(r["cost"] for r in g[1])):
        placed = ", ".join(f"{d} {n}" for d, n in Counter(r["device"] for r in group).most_common())
        supported = ", ".join(sorted({r["supported"] for r in group}))
        share = 100 * sum(r["cost"] for r in group) / total_cost
        print(f"  {op:24} {len(group):5}  {placed:18} {supported:18} {share:6.2f}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "midas_small_fp16"
    # The compiled copy is temporary and is deleted with the MLModel, so keep it alive
    model = ct.models.MLModel(str(MODELS_DIR / f"{name}.mlpackage"))
    compiled = model.get_compiled_model_path()
    print(f"{name}: Core ML's planned placement\n")

    plans = {units: plan_ops(compiled, units) for units in UNITS}
    for units, rows in plans.items():
        summarize(units, rows)

    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL):
        rows = plans[units]
        print(f"\n{units.name}, by op type:")
        by_op_type(rows)
        off_ane = [(i, r) for i, r in enumerate(rows) if r["device"] != "ANE"]
        print(f"\n  ops not on the ANE: {len(off_ane)}")
        for i, r in off_ane:
            print(f"    #{i:<4} {r['op']:20} on {r['device']}  (supports {r['supported']})  "
                  f"cost {100 * r['cost']:.3f}%  output {r['output']}")

    results = COREML_DIR / "results"
    results.mkdir(exist_ok=True)
    path = results / f"placement_{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "op", "output", "device", "supported", "cost"])
        writer.writeheader()
        for i, r in enumerate(plans[ct.ComputeUnit.CPU_AND_NE]):
            writer.writerow({"index": i, **r})
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
