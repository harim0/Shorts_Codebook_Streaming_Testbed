#!/usr/bin/env python3
"""
auto_dnn_mpd.py
클래스당 하나의 dnn.mpd를 data/{class}/ 아래에 생성한다.
개별 video MPD(data/{class}/{vid}/multi_resolution.mpd)에서는 DNN 노드를 제거한다.

Flow (NAS 방식과 동일):
  player detects new class
  → GET data/{class}/dnn.mpd
  → download decoder.pt + coarse_sr.pt from CDN
  → POST /dnn (Flask server loads model)
  → stream video segments (POST /uploader per segment)

Usage:
    python3 auto_dnn_mpd.py [data_root]
"""

import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

DATA_ROOT = "/home/harim/Shorts_Codebook_Streaming_Testbed/cdn-server/contentServer/dash/data"
DNN_HOST  = "http://163.152.162.202:8080"

MPD_NS = "urn:mpeg:dash:schema:mpd:2011"
NS     = {"mpd": MPD_NS}
ET.register_namespace("", MPD_NS)


def write_pretty_xml(root_elem, out_path):
    rough = ET.tostring(root_elem, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty)


def make_class_dnn_mpd(class_dir, class_name):
    """data/{class}/dnn.mpd 생성 — DNN 노드만 포함."""
    tag = lambda name: f"{{{MPD_NS}}}{name}"

    root = ET.Element(tag("MPD"))
    dnn_url = f"{DNN_HOST}/dash/model/{class_name}/"

    dnn = ET.SubElement(root, tag("DNN"), {"url": dnn_url})
    ET.SubElement(dnn, tag("Model"),     {"name": "decoder.pt"})
    ET.SubElement(dnn, tag("Model"),     {"name": "coarse_sr.pt"})
    ET.SubElement(dnn, tag("cluster"),   {"id": class_name})
    ET.SubElement(dnn, tag("frameRate"), {"fps": "30"})

    out_path = os.path.join(class_dir, "dnn.mpd")
    write_pretty_xml(root, out_path)
    print(f"  [DNN MPD] {out_path}")


def strip_dnn_from_video_mpd(mpd_path):
    """video MPD에서 DNN 노드 제거 (있으면)."""
    tree = ET.parse(mpd_path)
    root = tree.getroot()

    old = root.findall("mpd:DNN", NS)
    if not old:
        return
    for d in old:
        root.remove(d)

    rough = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    with open(mpd_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"  [CLEAN]  {mpd_path}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT

    if not os.path.isdir(base):
        print(f"Base dir not found: {base}")
        sys.exit(1)

    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n>>> class: {class_name}")

        # 1) 클래스 단위 dnn.mpd 생성
        make_class_dnn_mpd(class_dir, class_name)

        # 2) 개별 video MPD에서 DNN 노드 제거
        for vid_name in sorted(os.listdir(class_dir)):
            vid_dir = os.path.join(class_dir, vid_name)
            if not os.path.isdir(vid_dir):
                continue
            mpd_path = os.path.join(vid_dir, "multi_resolution.mpd")
            if os.path.exists(mpd_path):
                strip_dnn_from_video_mpd(mpd_path)


if __name__ == "__main__":
    main()
