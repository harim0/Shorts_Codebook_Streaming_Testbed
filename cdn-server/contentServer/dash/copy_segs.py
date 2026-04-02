import os
import shutil

src = "/home/harim/harim_SRSHORTS/cdn-server/contentServer/dash/data"
dst = "/home/harim/Dashlet_www/cdn-server/contentServer/dash/data"

extensions = [".m4s", ".mpd"]

for direc in os.listdir(src):
    src_pid_dir = os.path.join(src, direc)
    dst_pid_dir = os.path.join(dst, direc)

    # 폴더만 처리
    if not os.path.isdir(src_pid_dir):
        continue

    # 목적지 pid 폴더가 없으면 생성
    os.makedirs(dst_pid_dir, exist_ok=True)

    for filename in os.listdir(src_pid_dir):
        if any(filename.endswith(ext) for ext in extensions):
            src_file = os.path.join(src_pid_dir, filename)
            dst_file = os.path.join(dst_pid_dir, filename)

            shutil.copy(src_file, dst_file)
            print(f"Copied: {src_file} → {dst_file}")
