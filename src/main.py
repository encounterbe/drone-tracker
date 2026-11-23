#!/usr/bin/env python3
"""Command-line entry point.

Examples::

    # Stage 1 - analyse a recorded clip with the operator display
    python -m src.main --source output/test_drone.mp4

    # Headless batch run, writing an annotated video and a track log
    python -m src.main --source clip.mp4 --no-display \
        --record output/annotated.mp4 --export-tracks output/tracks.jsonl

    # Live USB camera (Stage 2)
    python -m src.main --source 0 --realtime

    # Tune a parameter without editing the config file
    python -m src.main --source clip.mp4 --set detection.motion.min_area=30
"""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import Pipeline
from .utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="uas-tracker",
        description="UAS detection and tracking with an operator HUD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--source", required=True,
        help="video file path, or a camera index such as 0",
    )
    ap.add_argument("--config", default=None, help="YAML config merged over the defaults")
    ap.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set ui.theme=nightvision (repeatable)",
    )
    ap.add_argument(
        "--backend", choices=["motion", "cnn", "fusion"], default=None,
        help="detection backend (overrides config)",
    )
    ap.add_argument("--weights", default=None, help="CNN weights path (implies --backend cnn)")
    ap.add_argument("--theme", default=None, help="HUD theme name")
    ap.add_argument("--record", default=None, help="write the annotated video here")
    ap.add_argument("--export-tracks", default=None, help="write per-frame track data as JSONL")
    ap.add_argument("--no-display", action="store_true", help="headless; no window")
    ap.add_argument("--realtime", action="store_true", help="throttle playback to source FPS")
    ap.add_argument("--loop", action="store_true", help="restart the clip when it ends")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    ap.add_argument("--zoom", type=float, default=None, help="initial digital zoom factor")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    # Flags win over config file values.
    if args.backend:
        cfg.detection["backend"] = args.backend
    if args.weights:
        cfg.detection["cnn"]["weights"] = args.weights
        if not args.backend:
            cfg.detection["backend"] = "cnn"
    if args.theme:
        cfg.ui["theme"] = args.theme
    if args.loop:
        cfg.video["loop"] = True
    if args.zoom is not None:
        cfg.zoom["initial"] = float(args.zoom)

    source: str | int = int(args.source) if str(args.source).isdigit() else args.source

    try:
        pipeline = Pipeline(
            source=source,
            cfg=cfg,
            display=not args.no_display,
            record_path=args.record,
            export_path=args.export_tracks,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            realtime=args.realtime,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Could not start pipeline: {exc}", file=sys.stderr)
        return 1

    try:
        result = pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(result.summary())
    if result.recording_path:
        print(f"Annotated video: {result.recording_path}")
    if result.export_path:
        print(f"Track export:    {result.export_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
