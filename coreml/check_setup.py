"""Confirms the environment and MiDaS load correctly. Run: python check_setup.py"""
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from midas_loader import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, load_midas_small

REPO = Path(__file__).resolve().parent.parent
DEMO_FRAME = 240  # a frame of the real scene, without title cards

for package in ("coremltools", "torch", "numpy", "onnxruntime"):
    print(f"{package:12} {version(package)}")

model = load_midas_small()
params = sum(p.numel() for p in model.parameters())
# Upstream defines this block but forward() never runs it, so conversion drops it
unused = sum(p.numel() for p in model.scratch.refinenet4.resConfUnit1.parameters())
print(f"\nMiDaS small loaded: {params / 1e6:.2f}M parameters, {(params - unused) / 1e6:.2f}M used by forward()")

video = cv2.VideoCapture(str(REPO / "docs" / "demo.mp4"))
video.set(cv2.CAP_PROP_POS_FRAMES, DEMO_FRAME)
ok, bgr = video.read()
assert ok, "could not read docs/demo.mp4"
bgr = cv2.resize(bgr, (640, 480))
resized = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# PyTorch has no built-in normalization, so it's applied here. The ONNX graph does it itself.
mean = np.array(IMAGENET_MEAN, dtype=np.float32)
std = np.array(IMAGENET_STD, dtype=np.float32)
pytorch_input = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1))[None]
with torch.no_grad():
    pytorch_depth = model(pytorch_input).squeeze().numpy()

session = ort.InferenceSession(str(REPO / "assets" / "models" / "midas.onnx"), providers=["CPUExecutionProvider"])
onnx_input = rgb.transpose(2, 0, 1)[None]
onnx_depth = session.run(None, {session.get_inputs()[0].name: onnx_input})[0].squeeze()

print(f"PyTorch output: shape {pytorch_depth.shape}, range {pytorch_depth.min():.1f} to {pytorch_depth.max():.1f}")
print(f"ONNX output:    shape {onnx_depth.shape}, range {onnx_depth.min():.1f} to {onnx_depth.max():.1f}")
pearson = np.corrcoef(pytorch_depth.ravel(), onnx_depth.ravel())[0, 1]
print(f"PyTorch vs ONNX Pearson on one frame: {pearson:.5f}")
print("\nSETUP OK")
