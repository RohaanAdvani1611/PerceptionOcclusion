"""Builds MiDaS v2.1 small from the pinned local copies, with no network access,
plus the Core ML wrapper and the frame loader every script shares."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

COREML_DIR = Path(__file__).resolve().parent
THIRD_PARTY = COREML_DIR / "third_party"
MIDAS_DIR = THIRD_PARTY / "MiDaS"
BACKBONE_DIR = THIRD_PARTY / "gen-efficientnet-pytorch"
WEIGHTS = THIRD_PARTY / "weights" / "midas_v21_small_256.pt"
DATA_DIR = COREML_DIR / "data"
MODELS_DIR = COREML_DIR / "models"

INPUT_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_midas_small():
    """Returns the model in eval mode. It expects ImageNet-normalized RGB input."""
    sys.path.insert(0, str(MIDAS_DIR))
    from midas.midas_net_custom import MidasNet_small

    # MiDaS fetches its backbone from GitHub's latest code unless redirected here
    hub_load = torch.hub.load

    def load_pinned_backbone(repo, model, *args, **kwargs):
        return hub_load(str(BACKBONE_DIR), model, *args, source="local", **kwargs)

    torch.hub.load = load_pinned_backbone
    try:
        model = MidasNet_small(
            str(WEIGHTS),
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
    finally:
        torch.hub.load = hub_load
    return model.eval()


class MidasForCoreML(torch.nn.Module):
    """MiDaS with normalization built in, shaped for Core ML image input and output.

    Input: 1x3x256x256 BGR in [0, 1]. Core ML's image input supplies this from a BGRA
    camera buffer by scaling with 1/255, which is the only step it can do, since it has
    one scale for all channels and ImageNet has three. Channel order and mean/std
    happen here, exactly once.
    Output: 1x1x256x256 relative inverse depth, the shape a grayscale image output needs.
    """

    def __init__(self, midas):
        super().__init__()
        self.midas = midas
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("inv_std", 1.0 / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, bgr):
        rgb = torch.flip(bgr, dims=[1])
        return self.midas((rgb - self.mean) * self.inv_std).unsqueeze(1)


def load_frames(set_name):
    """Returns [(name, 256x256 RGB uint8)] for coreml/data/<set_name>, resized the way
    the C++ pipeline does it: a bilinear stretch with cv::resize."""
    frames = []
    for path in sorted((DATA_DIR / set_name).glob("*.png")):
        bgr = cv2.imread(str(path))
        resized = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append((path.stem, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    assert frames, f"no frames in {DATA_DIR / set_name}"
    return frames


def to_wrapper_input(rgb):
    """256x256 RGB uint8 -> the 1x3x256x256 BGR [0, 1] tensor MidasForCoreML expects."""
    bgr = rgb[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1)))[None]
