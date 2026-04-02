# qoe_logger.py
import math
from collections import defaultdict

class QoELogger:
    def __init__(self, path, chunk_length=5.0, q_to_vmaf=None):
        self.path = path
        self.chunk_length = float(chunk_length)
        self.q_to_vmaf = q_to_vmaf or {}   # {q: vmaf}
        self.fd = open(self.path, "w", buffering=1)
        # 비디오별 수집 상태
        self.dl = defaultdict(lambda: {  # per vid
            "segs": {},      # seg_idx -> dict(q, size_bytes, start_ms, finish_ms)
            "first_req_ms": None,
            "duration": None
        })

    def close(self):
        self.fd.close()

    def on_download(self, event):
        """
        event keys (네 서버 포맷):
          - url -> vid는 네가 parse_vid()로 추출
          - lastRequest (seg index), lastquality (q), lastChunkSize(bytes)
          - lastChunkStartTime(ms), lastChunkFinishTime(ms), duration(sec)
        """
        from time import time
        vid = event.get('vid')  # 미리 parse_vid 해서 넣거나 직접 추출
        if vid is None:
            return
        d = self.dl[vid]
        si = int(event['lastRequest'])
        q = int(event['lastquality'])
        size_b = int(event['lastChunkSize'])
        st_ms = int(event['lastChunkStartTime'])
        fin_ms = int(event['lastChunkFinishTime'])
        d["segs"][si] = {"q": q, "size_b": size_b, "st_ms": st_ms, "fin_ms": fin_ms}
        d["duration"] = float(event.get('duration') or 0.0)
        d["first_req_ms"] = st_ms if d["first_req_ms"] is None else min(d["first_req_ms"], st_ms)

    def on_session_end(self, vid, watch_time_sec):
        """
        스와이프/끝까지 보기 등으로 '시청 종료'가 확정되면 호출.
        """
        if vid not in self.dl:
            return
        d = self.dl[vid]
        segs = d["segs"]
        if not segs:
            return
        T = max(0.0, float(d.get("duration") or 0.0))
        W = min(float(watch_time_sec), T if T > 0 else float('inf'))
        L = self.chunk_length

        # --- 스타트업 지연: 첫 세그 요청 시작 ~ 첫 세그 다운로드 완료 근사
        first_idx = min(segs.keys())
        startup_ms = None
        if d["first_req_ms"] is not None and "fin_ms" in segs.get(first_idx, {}):
            startup_ms = max(0, segs[first_idx]["fin_ms"] - d["first_req_ms"])
        startup_s = (startup_ms or 0)/1000.0

        # --- 재생 시뮬레이션으로 리버퍼링 근사
        # 초기 재생 가능 시각을 "첫 세그 완료"로 두고, 각 세그 i가
        # 해당 시각 + i*L까지 준비되었는지 확인
        base_play_ms = segs[first_idx]["fin_ms"]
        played_seg_count = int(math.floor(W / L + 1e-9))
        rebuf_s = 0.0
        last_ready_ms = base_play_ms
        for i in range(played_seg_count):
            if i not in segs:
                # 해당 세그를 끝내지 못했으면, 남은 구간을 전부 리버퍼로 본다.
                need_ms = base_play_ms + i*L*1000
                # 재생 도중 시청이 끝났다면 그만큼만
                break
            need_ms = base_play_ms + i*L*1000
            fin_ms = segs[i]["fin_ms"]
            if fin_ms > need_ms:
                rebuf_s += (fin_ms - need_ms)/1000.0
                base_play_ms += (fin_ms - need_ms)  # 타임라인 밀림

        # --- 평균 비트레이트/해상도(VMAF)
        kbps_list = []
        vmaf_list = []
        last_q = None
        q_switch = 0
        bytes_wasted = 0
        delivered_kbits_within_W = 0.0

        for i, info in sorted(segs.items()):
            size_kbits = 8.0 * info["size_b"] / 1000.0
            if i < played_seg_count:
                seg_start = i * L
                seg_dur_in_W = max(0.0, min(L, W - seg_start))
                frac = seg_dur_in_W / L if L > 0 else 1.0
                kbps_list.append((size_kbits / L) * frac if L > 0 else 0.0)
                delivered_kbits_within_W += size_kbits * frac
                if info["q"] in self.q_to_vmaf:
                    vmaf_list.append(float(self.q_to_vmaf[info["q"]]) * frac)
                if last_q is not None and info["q"] != last_q:
                    q_switch += abs(info["q"] - last_q)
                last_q = info["q"]
            else:
                bytes_wasted += info["size_b"]

        avg_kbps = sum(kbps_list)/max(1e-9, (W/L)) if kbps_list else 0.0
        avg_vmaf = (sum(vmaf_list)/max(1e-9, (W/L))) if vmaf_list else None
        served_ratio = ((W - rebuf_s) / W) if W > 0 else None  # 시청 시간 중 stall-free 비율
        stall_free = int(rebuf_s == 0)
        eff_kbps_wall = (delivered_kbits_within_W / max(W, 1e-9)) if W > 0 else 0.0

        line = {
            "vid": vid,
            "watch_time": round(W, 3),
            "duration": round(T, 3),
            "served_ratio": round(served_ratio, 4) if served_ratio is not None else None,
            "stall_free": stall_free,
            "startup_delay_s": round(startup_s, 3),
            "rebuffer_s": round(rebuf_s, 3),
            "avg_kbps": round(avg_kbps, 1),
            "eff_kbps_wall": round(eff_kbps_wall, 1),
            "avg_vmaf": round(avg_vmaf, 1) if avg_vmaf is not None else None,
            "bytes_wasted": int(bytes_wasted),
            "q_switch_sum": int(q_switch)
        }
        self.fd.write("[qoe]\t" + "\t".join(f"{k}={v}" for k,v in line.items()) + "\n")
        self.fd.flush()
        del self.dl[vid]