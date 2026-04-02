import xml.etree.ElementTree as ET
from xml.etree import ElementTree as etree
import os

NS = {"mpd": "urn:mpeg:dash:schema:mpd:2011"}
etree.register_namespace("", NS["mpd"])
dataDir = "./dash/data/"
fileList = os.listdir(dataDir)

print(fileList)

for folderName in fileList:
    if folderName[0] == ".":
        continue

    backmpd = dataDir + folderName + "/manifest-back.mpd"
    curmpd = dataDir + folderName + "/manifest.mpd"

    if os.path.exists(backmpd):
        os.system("rm "+backmpd)

    os.system("""mv %s %s"""%(curmpd, backmpd))

    tree = ET.parse(backmpd)
    root = tree.getroot()
    
    period = root.find("mpd:Period", NS)
    if period is None:
        os.rename(backmpd, curmpd)
        print(f"[SKIP] Period 없음: {folderName}")
        continue
    
    sets = period.findall("mpd:AdaptationSet", NS)
    video_sets = [s for s in sets if s.get("contentType") == "video"]
    
    for vs in video_sets:
        rep = vs.find("mpd:Representation", NS)
        if rep is None:
            continue
        segtpl = rep.find("mpd:SegmentTemplate", NS)
        timeline = segtpl.find("mpd:SegmentTimeline", NS)
        timescale = int(segtpl.get("timescale","1"))
        start = int(segtpl.get("startNumber","1"))

        total = 0
        for S in timeline.findall("mpd:S", NS):
            total += 1 + int(S.get("r","0"))

        rep_id = rep.get("id")
        bitrate_max = 0.0
        firstS = timeline.find("mpd:S", NS)
        chunk_len = int(firstS.get("d"))/timescale if firstS is not None else 1.0
        
        for num in range(start, start + total):
            f = os.path.join(dataDir, folderName, f"chunk-stream{rep_id}-{num:05d}.m4s")
            if not os.path.exists(f):
                break
            size = os.path.getsize(f)
            br = (size * 8) / max(chunk_len, 1e-6)
            if br > bitrate_max:
                bitrate_max = br

        if bitrate_max > 0:
            rep.set("bandwidth", str(int(bitrate_max)))
            
    if len(video_sets) > 1:
        target = video_sets[0]
        moved_sets = []
        for vs in video_sets[1:]:
            rep = vs.find("mpd:Representation", NS)
            if rep is not None:
                target.append(rep)
                moved_sets.append(vs)
        for vs in moved_sets:
            period.remove(vs)

    tmp = curmpd + ".new"
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, curmpd)
    print(f"[✔] flattened {folderName}")