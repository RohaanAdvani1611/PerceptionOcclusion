"""Saves webcam frames for W8A8 calibration and accuracy testing.

Capture one scene at a time, moving the laptop slowly while it records:
    python capture_frames.py calibration desk
    python capture_frames.py heldout kitchen
    python capture_frames.py --list

Each scene can belong to only one set, so test frames never overlap calibration frames.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parent / "data"
TARGETS = {"calibration": 150, "heldout": 100}
WIDTH, HEIGHT = 640, 480
SECONDS_BETWEEN_FRAMES = 0.4  # far enough apart that neighbouring frames differ


def scenes_in(set_name):
    folder = DATA / set_name
    return {p.name.rsplit("_", 1)[0] for p in folder.glob("*.png")} if folder.exists() else set()


def print_summary():
    for set_name, target in TARGETS.items():
        frames = len(list((DATA / set_name).glob("*.png")))
        scenes = ", ".join(sorted(scenes_in(set_name))) or "none"
        print(f"{set_name:12} {frames:4}/{target} frames, scenes: {scenes}")


def capture(set_name, scene, count, camera_index):
    other_set = next(s for s in TARGETS if s != set_name)
    if scene in scenes_in(other_set):
        sys.exit(f"Scene '{scene}' is already in {other_set}. Use a different scene for {set_name}.")
    if scene in scenes_in(set_name):
        sys.exit(f"Scene '{scene}' already exists in {set_name}. Choose a new name.")
    if "_" in scene or "/" in scene:
        sys.exit("Scene names can't contain '_' or '/'.")

    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {camera_index}. Check camera permission for your terminal app.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    folder = DATA / set_name
    folder.mkdir(parents=True, exist_ok=True)
    print("SPACE starts recording. Q stops. Move the laptop slowly while it records.")

    recording = False
    saved = 0
    next_save = 0.0
    try:
        while saved < count:
            ok, frame = camera.read()
            if not ok:
                sys.exit("The camera stopped delivering frames.")
            if frame.shape[:2] != (HEIGHT, WIDTH):
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            now = time.monotonic()
            if recording and now >= next_save:
                saved += 1
                cv2.imwrite(str(folder / f"{scene}_{saved:04}.png"), frame)
                next_save = now + SECONDS_BETWEEN_FRAMES

            preview = frame.copy()
            label = f"RECORDING {saved}/{count}" if recording else "Press SPACE to record"
            cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = True
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} frames to {folder}\n")
    print_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", nargs="?", choices=list(TARGETS), help="which set this scene belongs to")
    parser.add_argument("scene", nargs="?", help="short scene name, such as desk")
    parser.add_argument("--count", type=int, default=50, help="frames to save for this scene")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--list", action="store_true", help="show progress and exit")
    args = parser.parse_args()

    if args.list:
        print_summary()
    elif args.set and args.scene:
        capture(args.set, args.scene, args.count, args.camera)
    else:
        parser.error("give a set and a scene name, or use --list")


if __name__ == "__main__":
    main()
