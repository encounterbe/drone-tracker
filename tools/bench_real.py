#!/usr/bin/env python3
"""Run the real-footage benchmark and print one comparable table.

This is the real-footage counterpart to the synthetic regression suite in
docs/ARCHITECTURE.md section 5. Three clips in ``video/``, chosen because they fail in
three different ways, and a detection change that helps one can easily wreck
another:

``urban``   test.mp4, handheld pan over a street. **No airborne target at all**,
            so every confirmed track is provably false. This is the false-alarm
            measurement, and it needed no labelling.
``night``   test2.mp4, static CCTV at night. A bright ~10 px point target
            descending through heavy sensor noise. The real small-dim-target
            case.
``fpv``     test3.mp4, chase camera on an aerobatic model aircraft. Moving
            camera with hard roll, target crossing sky, sun flare and ground
            clutter.

Ground truth for the two target clips comes from ``tools/label_real.py`` and was
checked by eye against the contact sheets it emits - see the caveat there.

NOTE: the clips themselves are not in version control (``video/`` is gitignored,
and they are hundreds of megabytes). Without them this script reports each
scenario as missing. The numbers recorded in docs/ARCHITECTURE.md section 5b came from a
specific set of clips; swapping in different footage gives different numbers and
the table should be re-measured rather than compared.

Usage::

    python tools/bench_real.py
    python tools/bench_real.py --set detection.motion.edge_sensitivity=0.4
    python tools/bench_real.py --only night
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = {
    "urban": {
        "source": "video/test.mp4",
        "start": 0,
        "frames": 425,
        "fps": 29.97,
        "gt": None,  # negative clip
    },
    "night": {
        "source": "video/test2.mp4",
        "start": 1150,
        "frames": 500,
        "fps": 22.0,
        "gt": "output/test2_groundtruth.json",
    },
    "fpv": {
        "source": "video/test3.mp4",
        "start": 5250,
        "frames": 500,
        "fps": 60.0,
        "gt": "output/test3_groundtruth.json",
    },
}


def run(name: str, spec: dict, overrides: list[str], warmup: int) -> dict:
    tmp = os.path.join(tempfile.gettempdir(), f"bench_{name}.jsonl")
    cmd = [
        sys.executable, "-m", "src.main",
        "--source", spec["source"],
        "--no-display",
        "--start-frame", str(spec["start"]),
        "--max-frames", str(spec["frames"]),
        "--export-tracks", tmp,
    ]
    for o in overrides:
        cmd += ["--set", o]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ev = [sys.executable, "tools/eval_real.py", "--tracks", tmp,
          "--fps", str(spec["fps"]), "--warmup", str(warmup)]
    if spec["gt"]:
        ev += ["--ground-truth", spec["gt"]]
    else:
        ev += ["--expect-none"]
    out = subprocess.run(ev, cwd=ROOT, check=True, capture_output=True, text=True).stdout

    parsed: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line and not line.startswith("="):
            k, _, v = line.partition(":")
            parsed[k.strip()] = v.strip()
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override passed through to src.main (repeatable)")
    ap.add_argument("--only", default=None, help="run a single scenario by name")
    ap.add_argument("--warmup", type=int, default=35)
    args = ap.parse_args()

    names = [args.only] if args.only else list(SCENARIOS)
    results = {}
    for name in names:
        if name not in SCENARIOS:
            ap.error(f"unknown scenario {name!r}; pick from {list(SCENARIOS)}")
        clip = os.path.join(ROOT, SCENARIOS[name]["source"])
        if not os.path.exists(clip):
            print(f"skipping {name}: {SCENARIOS[name]['source']} not present", file=sys.stderr)
            continue
        print(f"running {name} ...", file=sys.stderr)
        results[name] = run(name, SCENARIOS[name], args.overrides, args.warmup)

    if not results:
        print("\nNo clips found under video/. This benchmark needs real footage;\n"
              "see docs/ARCHITECTURE.md section 5b for what the three clips are.")
        return 1

    if args.overrides:
        print("\noverrides: " + " ".join(args.overrides))

    print("\n--- target clips " + "-" * 45)
    print(f"{'scenario':<10} {'recall':>8} {'precis':>8} {'F1':>8} {'meanIoU':>8} {'IDsw':>6}")
    for name, r in results.items():
        if "recall" not in r:
            continue
        print(f"{name:<10} {r['recall']:>8} {r['precision']:>8} {r['f1']:>8} "
              f"{r['mean_iou']:>8} {r['id_switches']:>6}")

    print("\n--- negative clips (every contact is false) " + "-" * 14)
    print(f"{'scenario':<10} {'false trk/min':>14} {'false lock':>12} {'peak':>6} {'det/frame':>10}")
    for name, r in results.items():
        if "false_tracks_per_min" not in r:
            continue
        print(f"{name:<10} {r['false_tracks_per_min']:>14} {r['false_lock_rate']:>12} "
              f"{r['peak_concurrent_false']:>6} {r['detections_per_frame']:>10}")

    # No ground truth needed: a camera that is plainly moving, or plainly on a
    # tripod, should switch regime about once. More means the ego estimate is
    # oscillating and silently blinding the detector on every rebuild.
    print("\n--- camera-regime health " + "-" * 32)
    print(f"{'scenario':<10} {'mode switches':>14} {'blind':>8} {'ego unmeasured':>16}")
    for name, r in results.items():
        if "mode_switches" not in r:
            continue
        print(f"{name:<10} {r['mode_switches']:>14} {r['blind_frac']:>8} "
              f"{r['ego_unmeasured_frac']:>16}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
