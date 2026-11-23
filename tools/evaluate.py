#!/usr/bin/env python3
"""Score a track export against ground truth.

Detection work that is only ever eyeballed cannot be tuned: a parameter change
that fixes one clip and breaks two others looks like progress on screen. This
tool turns "does it work?" into numbers, so a config change can be justified.

Metrics
-------
``Recall``        fraction of ground-truth target frames where a confirmed track
                  sat within the match radius.
``Precision``     fraction of confirmed track observations that matched *some*
                  ground-truth object (target or distractor).
``ID switches``   how often the ID covering the target changed. The single most
                  useful tracker metric - a system that re-IDs every 20 frames
                  is useless for cueing a gimbal even at 100% recall.
``Mean IoU``      localisation quality on matched frames.
``Latency``       frames from the target first appearing to first confirmed lock.

Usage::

    python -m src.main --source output/test_drone.mp4 --no-display \\
        --export-tracks output/tracks.jsonl
    python tools/evaluate.py --tracks output/tracks.jsonl \\
        --ground-truth output/test_drone_groundtruth.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def evaluate(
    tracks_path: str,
    gt_path: str,
    target_class: str = "drone",
    warmup: int = 15,
    radius_factor: float = 1.5,
    min_radius: float = 25.0,
) -> dict:
    records = load_jsonl(tracks_path)
    with open(gt_path, "r", encoding="utf-8") as fh:
        gt_meta = json.load(fh)
    gt_by_frame = {int(f["frame"]): f["objects"] for f in gt_meta["frames_data"]}

    matched_frames = 0
    evaluated_frames = 0
    ious: list[float] = []
    covering_ids: list[int | None] = []
    id_switches = 0
    first_lock: int | None = None
    total_track_obs = 0
    matched_track_obs = 0

    for record in records:
        frame = int(record["frame"])
        objects = gt_by_frame.get(frame, [])
        targets = [o for o in objects if o["class"] == target_class]
        confirmed = [t for t in record["tracks"] if t["state"] in ("CONFIRMED", "COASTING")]

        # --- precision side: does each reported track correspond to anything real?
        for track in confirmed:
            total_track_obs += 1
            tc = center(track["bbox"])
            for obj in objects:
                oc = center(obj["bbox"])
                radius = max(min_radius, radius_factor * float(obj.get("size_px", 20)))
                if np.hypot(tc[0] - oc[0], tc[1] - oc[1]) <= radius:
                    matched_track_obs += 1
                    break

        if frame < warmup or not targets:
            continue

        evaluated_frames += 1
        target = targets[0]
        gc = center(target["bbox"])
        radius = max(min_radius, radius_factor * float(target.get("size_px", 20)))

        best_id, best_iou, best_dist = None, 0.0, float("inf")
        for track in confirmed:
            tc = center(track["bbox"])
            dist = float(np.hypot(tc[0] - gc[0], tc[1] - gc[1]))
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best_id = int(track["id"])
                best_iou = iou(track["bbox"], target["bbox"])

        if best_id is not None:
            matched_frames += 1
            ious.append(best_iou)
            if first_lock is None:
                first_lock = frame
        covering_ids.append(best_id)

    # Count ID switches across contiguous covered stretches.
    previous = None
    for tid in covering_ids:
        if tid is None:
            continue
        if previous is not None and tid != previous:
            id_switches += 1
        previous = tid

    recall = matched_frames / evaluated_frames if evaluated_frames else 0.0
    precision = matched_track_obs / total_track_obs if total_track_obs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "frames_evaluated": evaluated_frames,
        "frames_matched": matched_frames,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "id_switches": id_switches,
        "unique_ids_on_target": len({i for i in covering_ids if i is not None}),
        "first_lock_frame": first_lock,
        "acquisition_latency_frames": (first_lock - warmup) if first_lock is not None else None,
        "track_observations": total_track_obs,
        "track_observations_matched": matched_track_obs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score tracking output against ground truth.")
    ap.add_argument("--tracks", required=True, help="JSONL produced by --export-tracks")
    ap.add_argument("--ground-truth", required=True, help="*_groundtruth.json from make_test_video")
    ap.add_argument("--class", dest="target_class", default="drone")
    ap.add_argument("--warmup", type=int, default=15, help="frames to skip (background model warm-up)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = ap.parse_args()

    result = evaluate(args.tracks, args.ground_truth, args.target_class, args.warmup)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 62)
    print("TRACKING EVALUATION")
    print("=" * 62)
    print(f"  Frames evaluated        {result['frames_evaluated']}")
    print(f"  Frames with target hit  {result['frames_matched']}")
    print(f"  Recall                  {result['recall'] * 100:6.2f} %")
    print(f"  Precision               {result['precision'] * 100:6.2f} %")
    print(f"  F1                      {result['f1'] * 100:6.2f} %")
    print(f"  Mean IoU (matched)      {result['mean_iou']:6.3f}")
    print(f"  ID switches on target   {result['id_switches']}")
    print(f"  Unique IDs on target    {result['unique_ids_on_target']}")
    print(f"  Acquisition latency     {result['acquisition_latency_frames']} frames")
    print("=" * 62)


if __name__ == "__main__":
    main()
