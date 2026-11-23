#!/usr/bin/env python3
"""Run the three synthetic regression scenarios from docs/ARCHITECTURE.md section 5.

The real-footage benchmark (``tools/bench_real.py``) measures whether the system
works on real video. This one measures whether a change has quietly broken the
cases that were already working. Both matter: the small-target work in this
project's history was justified precisely by a change that helped the hard clip
*and* did not cost the static one.

Expects the clips to exist already::

    python tools/make_test_video.py --out output/test_drone.mp4 --seconds 20
    python tools/make_test_video.py --out output/test_pan.mp4 --seconds 12 --camera-pan 0.8
    python tools/make_test_video.py --out output/test_hard.mp4 --seconds 16 \\
        --target-blur 1.8 --haze 0.62 --size-range 8 14 --no-bird

Usage::

    python tools/bench_synthetic.py
    python tools/bench_synthetic.py --set detection.motion.edge_sensitivity=0.25
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = {
    "static": "output/test_drone.mp4",
    "pan": "output/test_pan.mp4",
    "hard": "output/test_hard.mp4",
}


def run(name: str, clip: str, overrides: list[str], warmup: int) -> dict:
    tmp = os.path.join(tempfile.gettempdir(), f"syn_{name}.jsonl")
    cmd = [sys.executable, "-m", "src.main", "--source", clip,
           "--no-display", "--export-tracks", tmp]
    for o in overrides:
        cmd += ["--set", o]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gt = os.path.splitext(clip)[0] + "_groundtruth.json"
    ev = [sys.executable, "tools/evaluate.py", "--tracks", tmp,
          "--ground-truth", gt, "--warmup", str(warmup), "--json"]
    out = subprocess.run(ev, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _num(value) -> str:
    if isinstance(value, float):
        return f"{value * 100:.1f} %" if value <= 1.0 else f"{value:.3f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--warmup", type=int, default=35)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    names = [args.only] if args.only else list(SCENARIOS)
    if args.overrides:
        print("overrides: " + " ".join(args.overrides))
    print(f"{'scenario':<9} {'recall':>9} {'precision':>10} {'F1':>8} {'meanIoU':>9} {'IDsw':>6}")
    for name in names:
        clip = SCENARIOS[name]
        if not os.path.exists(os.path.join(ROOT, clip)):
            print(f"{name:<9}  missing {clip} - generate it first")
            continue
        r = run(name, clip, args.overrides, args.warmup)
        print(f"{name:<9} {_num(r.get('recall')):>9} {_num(r.get('precision')):>10} "
              f"{_num(r.get('f1')):>8} {_num(r.get('mean_iou')):>9} {str(r.get('id_switches')):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
