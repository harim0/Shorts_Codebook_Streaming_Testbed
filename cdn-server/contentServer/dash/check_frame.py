import os, glob

# ✅ PID와 기본 경로를 여기만 바꾸면 됩니다
pid = "100712"
base = f"/home/harim/harim_SRSHORTS/cdn-server/contentServer/dash/data/{pid}"
fps = "60fps"

# 확인할 해상도 목록
res_list = ["240p", "360p", "480p", "720p", "1080p"]

def count_png(folder):
    return len(glob.glob(os.path.join(folder, "*.png")))

print(f"=== Checking frame counts for PID {pid} ===\n")

hr_dir = os.path.join(base, f"1080p-{fps}")
hr_count = count_png(hr_dir)
print(f"[HR] {hr_dir}: {hr_count}")

for res in res_list:
    if res == "1080p":
        continue
    dirs = [
        os.path.join(base, f"{res}-{fps}"),
        os.path.join(base, f"{res}-{fps}-no-upscale"),
    ]
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[{res}] ❌ Missing folder: {d}")
            continue
        c = count_png(d)
        mark = "✅" if c == hr_count else "❌"
        print(f"[{res}] {mark} {d}: {c} frames (vs HR {hr_count})")

print("\nDone.")
