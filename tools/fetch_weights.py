#!/usr/bin/env python3
"""Download pre-trained UAS detection weights.

The weights are third-party, over 100 MB, and not in version control. A script
records *which* file was fetched from *where* and checks it arrived intact,
keeping the provenance next to the code instead of in someone's shell history.

Nothing here installs anything. Running the downloaded weights additionally
needs ``ultralytics`` (which pulls in PyTorch):

    pip install ultralytics

Both models are trained on the Kaggle "YOLO Drone Detection" dataset, which is
dominated by *close* drones. Expect them to do well on a drone filling a good
part of the frame and poorly on the 10-20 px distant target that the motion
path exists for. That is why the recommended backend is ``fusion``, not ``cnn``.

Usage::

    python tools/fetch_weights.py --list
    python tools/fetch_weights.py --model yolov11x
    python -m src.main --source clip.mp4 --config config/drone_yolo.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(ROOT, "weights")

MODELS = {
    "yolov8x": {
        "url": "https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x/resolve/main/weight/best.pt",
        "filename": "drone_yolov8x.pt",
        "approx_mb": 137,
        "source": "https://github.com/doguilmak/Drone-Detection-YOLOv8x (MIT)",
        "note": "Largest YOLOv8. ~1-2 s/frame on CPU - fine offline, unusable live.",
    },
    "yolov11x": {
        "url": "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt",
        "filename": "drone_yolov11x.pt",
        "approx_mb": 114,
        "source": "https://github.com/doguilmak/Drone-Detection-YOLOv11x (MIT)",
        "note": "Newer architecture, same dataset. Try this one first.",
    },
}


def _progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stderr.write(f"\r  {done / 1e6:6.1f} MB")
        return
    pct = 100.0 * done / total
    bar = "#" * int(pct // 4)
    sys.stderr.write(f"\r  [{bar:<25}] {pct:5.1f} %  {done / 1e6:6.1f}/{total / 1e6:.1f} MB")


def download(key: str, force: bool) -> int:
    spec = MODELS[key]
    os.makedirs(DEST_DIR, exist_ok=True)
    dest = os.path.join(DEST_DIR, spec["filename"])

    if os.path.exists(dest) and not force:
        size = os.path.getsize(dest) / 1e6
        print(f"already present: {dest} ({size:.0f} MB) - use --force to re-download")
        return 0

    print(f"source : {spec['source']}")
    print(f"target : {dest}")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
        sys.stderr.write("\n")
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    # A truncated download is worse than none: it fails later inside the model
    # loader with an error that looks like corrupt weights, not a network problem.
    got_mb = os.path.getsize(tmp) / 1e6
    if total and abs(got_mb - total / 1e6) > 0.5:
        os.remove(tmp)
        print(f"size mismatch ({got_mb:.1f} MB of {total / 1e6:.1f} MB) - discarded", file=sys.stderr)
        return 1

    os.replace(tmp, dest)
    print(f"done: {dest} ({got_mb:.0f} MB)")
    print()
    print("Next (needs `pip install ultralytics`):")
    print("  python -m src.main --source clip.mp4 --config config/drone_yolo.yaml \\")
    print(f"      --set detection.cnn.weights=weights/{spec['filename']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), help="which weights to fetch")
    ap.add_argument("--list", action="store_true", help="show the available models and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.list or not args.model:
        print("Available pre-trained UAS weights:\n")
        for key, spec in sorted(MODELS.items()):
            print(f"  {key:<10} ~{spec['approx_mb']:>4} MB   {spec['filename']}")
            print(f"  {'':<10} {spec['source']}")
            print(f"  {'':<10} {spec['note']}\n")
        print("Fetch one with:  python tools/fetch_weights.py --model yolov11x")
        return 0

    return download(args.model, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
