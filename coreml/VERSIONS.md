# Pinned versions

Python packages are pinned in `requirements.txt` (Python 3.12.7). The downloads below live in `third_party/`, which isn't tracked by git.

| Item | Version | Source |
|---|---|---|
| MiDaS source | `454597711a62eabcbf7d1e89f3fb9f569051ac9b` | https://github.com/isl-org/MiDaS |
| EfficientNet backbone source | `771ce082b2ce6d033f55b3d47c1f77389ad3c180` | https://github.com/rwightman/gen-efficientnet-pytorch |
| `weights/midas_v21_small_256.pt` | SHA-256 `70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca` | https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt |

## To recreate

```bash
cd coreml
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

mkdir -p third_party/weights
git clone https://github.com/isl-org/MiDaS.git third_party/MiDaS
git -C third_party/MiDaS checkout 454597711a62eabcbf7d1e89f3fb9f569051ac9b
git clone https://github.com/rwightman/gen-efficientnet-pytorch.git third_party/gen-efficientnet-pytorch
git -C third_party/gen-efficientnet-pytorch checkout 771ce082b2ce6d033f55b3d47c1f77389ad3c180
curl -L -o third_party/weights/midas_v21_small_256.pt https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
shasum -a 256 third_party/weights/midas_v21_small_256.pt
```
