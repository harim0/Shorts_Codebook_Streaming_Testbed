#!/usr/bin/env python3
"""
gen_sequence.py
data/{class}/*.mp4 를 순회해 ffprobe로 duration 추출 후
web-server/static/sequence.json 을 생성한다.

Entry format:
  {"pid": "Animal/SDR_Animal_23rp", "class": "Animal",
   "duration": 55.3, "watch_time": 41.5}

watch_time = duration * 0.75 (Shorts 특성: 끝까지 보지 않고 스와이프)
"""

import os
import json
import subprocess

DATA_ROOT   = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH    = os.path.join(os.path.dirname(__file__),
                           "../../../web-server/static/sequence.json")


def get_duration(mp4_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet",
         "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         mp4_path],
        capture_output=True, text=True
    )
    val = result.stdout.strip()
    if val and val != "N/A":
        return float(val)

    # fallback: format-level duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         mp4_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def main():
    entries = []

    for class_name in sorted(os.listdir(DATA_ROOT)):
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith(".mp4"):
                continue

            vid = fname[:-4]  # strip .mp4
            mp4_path = os.path.join(class_dir, fname)

            duration = get_duration(mp4_path)
            watch_time = round(duration * 0.75, 3)

            entries.append({
                "pid":        f"{class_name}/{vid}",
                "class":      class_name,
                "duration":   round(duration, 3),
                "watch_time": watch_time,
            })
            print(f"  {class_name}/{vid}  dur={duration:.3f}s  watch={watch_time:.3f}s")

    out_path = os.path.normpath(OUT_PATH)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
