#!/usr/bin/env python3
"""
batch_process_videos.py
───────────────────────
Processes all videos in:
  deliverables/stage_a_256_13kp/demo/collection_defferent_videos/original
Using:
  1. stage-a-256-13kp-fp16.onnx -> output/output_fb16/
  2. stage-a-256-13kp-int8.onnx -> output/output_int8/
"""

import os
import sys
import time
from pathlib import Path

# Add demo directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from run_video_demo import run_demo

ORIGINAL_DIR = DEMO_DIR / "collection_defferent_videos" / "original"
OUTPUT_BASE = DEMO_DIR / "collection_defferent_videos" / "output"

MODELS = [
    {
        "name": "FP16",
        "weights": PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "models" / "stage-a-256-13kp-fp16.onnx",
        "output_dir": OUTPUT_BASE / "output_fb16"
    },
    {
        "name": "INT8",
        "weights": PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "models" / "stage-a-256-13kp-int8.onnx",
        "output_dir": OUTPUT_BASE / "output_int8"
    }
]

def main():
    video_files = sorted(list(ORIGINAL_DIR.glob("*.mp4")))
    print(f"Found {len(video_files)} videos in {ORIGINAL_DIR}")

    t_total_start = time.time()

    for m in MODELS:
        model_name = m["name"]
        weights_path = m["weights"]
        out_dir = m["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"STARTING BATCH PROCESSING WITH {model_name} MODEL: {weights_path.name}")
        print(f"Destination folder: {out_dir}")
        print("=" * 70)

        for i, vid_path in enumerate(video_files, 1):
            out_vid = out_dir / vid_path.name
            print(f"\n[{i}/{len(video_files)}] Processing {vid_path.name} with {model_name}...")
            t0 = time.time()
            try:
                run_demo(
                    video_path=str(vid_path),
                    weights_path=str(weights_path),
                    output_path=str(out_vid),
                    conf_thresh=0.25,
                    nms_thresh=0.45,
                    apply_smoothing=True,
                    expand_15kp=False
                )
                dur = time.time() - t0
                print(f"  [OK] Completed {vid_path.name} ({dur:.1f}s) -> Size: {out_vid.stat().st_size / 1e6:.1f} MB")
            except Exception as e:
                print(f"  [ERROR] Failed to process {vid_path.name}: {e}")

    total_dur = time.time() - t_total_start
    print("\n" + "=" * 70)
    print(f"[SUCCESS] ALL VIDEOS PROCESSED FOR FP16 AND INT8 IN {total_dur/60:.1f} MINUTES!")
    print("=" * 70)

if __name__ == "__main__":
    main()

