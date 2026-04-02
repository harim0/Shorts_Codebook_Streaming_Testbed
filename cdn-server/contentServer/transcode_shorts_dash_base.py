#!/usr/bin/env python3
import os
import json
import subprocess
import shlex

FFMPEG  = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

# SEQUENCE_JSON = "/home/harim/harim_SRSHORTS/web-server/static/sequence.json"
# INPUT_DIR     = "/home/harim/shorts/www/raw_file"
# OUTPUT_ROOT   = "./dash/data"
DATA_ROOT   = "/home/harim/Dashlet_www/cdn-server/contentServer/dash/data"
CHUNK_SECONDS = 4

# traditional (warning -> X guarantee SAR 1:1)
# LADDER = [
#     {"w": -2, "h": 240,  "b": "400k",  "maxrate":"600k",  "bufsize":"1200k"},
#     {"w": -2, "h": 360,  "b": "800k",  "maxrate":"1200k", "bufsize":"2400k"},
#     {"w": -2, "h": 480,  "b": "1200k", "maxrate":"1800k","bufsize":"3600k"},
#     {"w": -2, "h": 720,  "b": "2400k", "maxrate":"3600k","bufsize":"7200k"},
#     {"w": -2, "h": 1080, "b": "4800k", "maxrate":"7200k","bufsize":"14400k"},
# ]

LADDER = [
    {"w": 240,  "h": 426,  "b": "400k",  "maxrate":"600k",  "bufsize":"1200k"},
    {"w": 360,  "h": 640,  "b": "800k",  "maxrate":"1200k", "bufsize":"2400k"},
    {"w": 480,  "h": 854,  "b": "1200k", "maxrate":"1800k","bufsize":"3600k"},
    {"w": 720,  "h": 1280, "b": "2400k", "maxrate":"3600k","bufsize":"7200k"},
    {"w": 1080, "h": 1920, "b": "4800k", "maxrate":"7200k","bufsize":"14400k"},
]

def probe_fps(path):
    import json, subprocess
    info = json.loads(subprocess.check_output([
        FFPROBE,"-v","error","-select_streams","v:0",
        "-of","json","-show_streams", path
    ]).decode())
    r = info["streams"][0].get("r_frame_rate","30/1")
    num, den = map(float, r.split("/"))
    return num/den if den else 30.0

def run_ffmpeg(pid: str, in_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.mpd")
    if os.path.exists(manifest):
        print(f"[SKIP] {pid} already converted: {manifest}")
        return

    N = len(LADDER)
    split_labels = "".join(f"[v{i}]" for i in range(N))
    graph = (
        f"[0:v]setpts=PTS-STARTPTS,split={N}{split_labels};" +
        "".join(
            f"[v{i}]scale={r['w']}:{r['h']},setsar=1[v{i}s];"
            for i, r in enumerate(LADDER)
        ) +
        "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[aout]"
    )


    # FPS→GOP(4초) 계산
    fps  = probe_fps(in_path)
    gop  = max(1, int(round(fps * CHUNK_SECONDS)))  # CHUNK_SECONDS=4

    cmd = [
        FFMPEG, "-i", in_path,
        "-filter_complex", graph,
        "-pix_fmt","yuv420p",
        "-vsync","cfr","-r", str(int(round(fps))),
    ]

    for i, rung in enumerate(LADDER):
        cmd += [
            "-map", f"[v{i}s]",
            f"-c:v:{i}", "libx264",   
            f"-profile:v:{i}", "high",
            f"-b:v:{i}", rung["b"],
            f"-maxrate:v:{i}", rung["maxrate"],
            f"-bufsize:v:{i}", rung["bufsize"],
            f"-g:v:{i}", str(gop),
            f"-keyint_min:v:{i}", str(gop),
            f"-sc_threshold:v:{i}", "0",
            f"-force_key_frames:v:{i}", f"expr:gte(t,n_forced*{CHUNK_SECONDS})",
        ]

    cmd += [
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "44100",
        "-f", "dash", "-use_template", "1", "-use_timeline", "1",
        "-seg_duration", str(CHUNK_SECONDS),
        "-init_seg_name", "init-stream$RepresentationID$.m4s",
        "-media_seg_name", "chunk-stream$RepresentationID$-$Number%05d$.m4s",
        "-adaptation_sets", "id=0,streams=v id=1,streams=a",
        manifest
    ]

    print(">>>"," ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    print(f"[✔] {pid} 변환 완료: {manifest}")

def main():
    pids = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    print(f"[INFO] 대상 pid 개수: {len(pids)}")

    for pid in sorted(pids):
        in_dir = os.path.join(DATA_ROOT, pid)
        src = os.path.join(in_dir, f"{pid}.mp4")
        if not os.path.exists(src):
            print(f"[MISS] 입력 파일 없음: {src}")
            continue

        out_dir = in_dir  # 같은 폴더 안에 manifest/chunk들 생성
        try:
            run_ffmpeg(pid, src, out_dir)
        except subprocess.CalledProcessError as e:
            print(f"[ERR ] ffmpeg 실패 pid={pid}: {e}")

if __name__ == "__main__":
    main()

# def main():
#     with open(SEQUENCE_JSON, "r", encoding="utf-8") as f:
#         seq = json.load(f)
#     pids = {str(item["pid"]) for item in seq if "pid" in item}
#     print(f"[INFO] 대상 pid 개수: {len(pids)}")

#     for pid in sorted(pids):
#         src = os.path.join(INPUT_DIR, f"{pid}.mp4")
#         if not os.path.exists(src):
#             print(f"[MISS] 입력 파일 없음: {src}")
#             continue
#         out_dir = os.path.join(OUTPUT_ROOT, pid)
#         try:
#             run_ffmpeg(pid, src, out_dir)
#         except subprocess.CalledProcessError as e:
#             print(f"[ERR ] ffmpeg 실패 pid={pid}: {e}")

# if __name__ == "__main__":
#     main()
