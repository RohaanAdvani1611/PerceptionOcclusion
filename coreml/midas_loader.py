"""Builds MiDaS v2.1 small from the pinned local copies, with no network access."""
import sys
from pathlib import Path

import torch

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
MIDAS_DIR = THIRD_PARTY / "MiDaS"
BACKBONE_DIR = THIRD_PARTY / "gen-efficientnet-pytorch"
WEIGHTS = THIRD_PARTY / "weights" / "midas_v21_small_256.pt"

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
