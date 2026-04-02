#!/usr/bin/env python3
"""
inject_dnn_into_video_mpd.py

각 video MPD(data/{class}/{vid}/multi_resolution.mpd)에 다시 <DNN ...> 노드를 삽입한다.
모델 파일은 계속 클래스 폴더(dash/model/{class}/decoder.pt, coarse_sr.pt)를 공유한다.

Usage:
    python3 inject_dnn_into_video_mpd.py [data_root]
"""

import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

DATA_ROOT = "/home/harim/Shorts_Codebook_Streaming_Testbed/cdn-server/contentServer/dash/data"
DNN_HOST  = "http://163.152.162.202:8080"

MPD_NS = "urn:mpeg:dash:schema:mpd:2011"
NS = {"mpd": MPD_NS}
ET.register_namespace("", MPD_NS)


def tag(name: str) -> str:
    return f"{{{MPD_NS}}}{name}"


def pretty_write(root: ET.Element, out_path: str) -> None:
    rough = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty)


def infer_fps_from_mpd(root: ET.Element) -> str:
    # 1) DNN/frameRate가 이미 있으면 유지
    old_dnn = root.find("mpd:DNN", NS)
    if old_dnn is not None:
        fr = old_dnn.find("mpd:frameRate", NS)
        if fr is not None and fr.get("fps"):
            return fr.get("fps")

    # 2) Representation frameRate에서 추출
    rep = root.find(".//mpd:Representation", NS)
    if rep is not None:
        frame_rate = rep.get("frameRate")
        if frame_rate:
            if "/" in frame_rate:
                try:
                    num, den = frame_rate.split("/")
                    fps = round(float(num) / float(den))
                    return str(int(fps))
                except Exception:
                    pass
            try:
                return str(int(float(frame_rate)))
            except Exception:
                pass

    # 3) fallback
    return "30"


def upsert_dnn_node(mpd_path: str, class_name: str) -> None:
    tree = ET.parse(mpd_path)
    root = tree.getroot()

    # 기존 DNN 제거
    for dnn in root.findall("mpd:DNN", NS):
        root.remove(dnn)

    fps = infer_fps_from_mpd(root)
    dnn_url = f"{DNN_HOST}/dash/model/{class_name}/"

    # MPD 루트 바로 아래에 삽입
    dnn = ET.Element(tag("DNN"), {"url": dnn_url})
    ET.SubElement(dnn, tag("Model"), {"name": "decoder.pt"})
    ET.SubElement(dnn, tag("Model"), {"name": "coarse_sr.pt"})
    ET.SubElement(dnn, tag("cluster"), {"id": class_name})
    ET.SubElement(dnn, tag("frameRate"), {"fps": fps})

    # 가능하면 ProgramInformation 뒤, 없으면 맨 앞쪽에 삽입
    inserted = False
    children = list(root)
    for idx, child in enumerate(children):
        if child.tag == tag("ProgramInformation"):
            root.insert(idx + 1, dnn)
            inserted = True
            break

    if not inserted:
        root.insert(0, dnn)

    pretty_write(root, mpd_path)
    print(f"[OK] injected DNN into {mpd_path}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT

    if not os.path.isdir(base):
        print(f"[ERR] base dir not found: {base}")
        sys.exit(1)

    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n>>> class: {class_name}")

        for vid_name in sorted(os.listdir(class_dir)):
            vid_dir = os.path.join(class_dir, vid_name)
            if not os.path.isdir(vid_dir):
                continue

            mpd_path = os.path.join(vid_dir, "multi_resolution.mpd")
            if not os.path.exists(mpd_path):
                continue

            try:
                upsert_dnn_node(mpd_path, class_name)
            except Exception as e:
                print(f"[ERR] {mpd_path}: {e}")


if __name__ == "__main__":
    main()