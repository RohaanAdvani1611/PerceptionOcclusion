"""Saves the 100 held-out frames, resized to 256x256, into the DepthCam app for its benchmark mode.
    python export_bench_frames.py

Uses the same resize as every other script (load_frames), so the app times the models on
exactly the frames evaluate.py measured. The folder is gitignored: these are your photos.
"""
import cv2

from midas_loader import COREML_DIR, load_frames

OUT = COREML_DIR / "DepthCam" / "DepthCam" / "BenchFrames"


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("bench_*.png"):
        old.unlink()
    frames = load_frames("heldout")
    for name, rgb in frames:
        cv2.imwrite(str(OUT / f"bench_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Wrote {len(frames)} frames to {OUT}")


if __name__ == "__main__":
    main()
