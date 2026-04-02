import os, sys, xml.etree.ElementTree as ET
NS = {'mpd':'urn:mpeg:dash:schema:mpd:2011'}

def val_one(folder):
    mpd = os.path.join(folder, 'manifest.mpd')
    if not os.path.exists(mpd):
        return (False, f'[MISS] {mpd}')
    root = ET.parse(mpd).getroot()
    per = root.find('mpd:Period', NS);  assert per is not None, 'Period 없음'
    sets = per.findall('mpd:AdaptationSet', NS)
    vids = [s for s in sets if s.get('contentType')=='video']
    assert len(vids)==1, f'video AdaptationSet {len(vids)}개(1이어야함)'

    reps = vids[0].findall('mpd:Representation', NS)
    assert len(reps)>=2, f'video Representation {len(reps)}개(2+ 필요)'
    # SegmentTimeline 동형성 체크
    def sig(rep):
        tpl = rep.find('mpd:SegmentTemplate', NS);  assert tpl is not None, 'SegmentTemplate 없음'
        tl  = tpl.find('mpd:SegmentTimeline', NS);  assert tl is not None, 'SegmentTimeline 없음'
        ts  = int(tpl.get('timescale','1'))
        seq = [(int(s.get('d')), int(s.get('r','0'))) for s in tl.findall('mpd:S', NS)]
        return ts, tuple(seq)
    base_ts, base_seq = sig(reps[0])
    for rep in reps[1:]:
        ts, seq = sig(rep)
        assert ts==base_ts and seq==base_seq, '세 Representation 간 SegmentTimeline 불일치'

    # 파일 존재/매칭 체크(앞뒤 몇 개만 샘플)
    tpl = reps[0].find('mpd:SegmentTemplate', NS)
    tl = tpl.find('mpd:SegmentTimeline', NS)
    start = int(tpl.get('startNumber','1'))
    total = sum(1+int(s.get('r','0')) for s in tl.findall('mpd:S', NS))
    ids = [r.get('id') for r in reps]
    d = os.path.dirname(mpd)
    def must(p): 
        if not os.path.exists(p): raise AssertionError(f'파일없음: {p}')
    for rep_id in ids:
        for num in [start, start+1, start+total-1]:  # 샘플 3개
            must(os.path.join(d, f'chunk-stream{rep_id}-{num:05d}.m4s'))
        must(os.path.join(d, f'init-stream{rep_id}.m4s'))

    # media 템플릿 형태
    for rep in reps:
        media = rep.find('mpd:SegmentTemplate', NS).get('media','')
        assert '$RepresentationID$' in media, 'SegmentTemplate@media에 $RepresentationID$ 누락'

    return (True, f'OK: Q={len(reps)} S={total}')

if __name__=='__main__':
    base = sys.argv[1] if len(sys.argv)>1 else './dash/data'
    bad=[]
    for name in sorted(os.listdir(base)):
        path=os.path.join(base,name)
        if not os.path.isdir(path): continue
        try:
            ok,msg = val_one(path)
            print(f'[{name}] {msg}')
            if not ok: bad.append(name)
        except AssertionError as e:
            print(f'[{name}] FAIL: {e}')
            bad.append(name)
    if bad:
        sys.exit(1)
