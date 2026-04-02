#!/usr/bin/env python3
import os, sys, json, argparse, copy, math, itertools, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

# ======== 환경/상수 ========
CHUNK_LENGTH = 4.0  # 초 단위 세그 길이(대부분 5s)
LOG_FILE = "/home/harim/Dashlet_for_www/data/exp/qoe_out.log"

# ======== 유틸 ========
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def parse_vid(url: str) -> str:
    try:
        return url.split('/')[5]
    except Exception:
        return None

def validate_chunksizes(sizes):
    """
    기대 형태: sizes = [S][Q]  (S=세그먼트 수, Q=품질 수), 값 단위=바이트
    직사각 2차원 배열만 통과. 아니면 [] 반환해 상위에서 -2를 내보내게 함.
    """
    if not isinstance(sizes, list) or not sizes: return []
    if not isinstance(sizes[0], list): return []
    rowlen = len(sizes[0])
    if any((not isinstance(r, list) or len(r) != rowlen) for r in sizes):
        return []
    return sizes

def normalize_event_inplace(ev: dict):
    ev['chunksize_list'] = validate_chunksizes(ev.get('chunksize_list'))

def bytes_to_kbits(x_bytes: float) -> float:
    # kbits = bytes * 8 / 1000  (KB가 아니라 '킬로비트' 스케일을 맞춤)
    return float(x_bytes) * 8.0 / 1000.0

def get_bitrate_kbits_sq(events):
    """
    각 이벤트의 chunksize_list(바이트)를 kbits로 변환해 [S][Q] 유지
    return: list of [S][Q] (kbits)
    """
    out = []
    for ev in events:
        sq = validate_chunksizes(ev.get('chunksize_list'))
        out.append([[bytes_to_kbits(sz) for sz in row] for row in sq])
    return out

# ======== 스루풋 추정 (최근 5개 조화평균) ========
class ThroughputEstimator:
    def __init__(self):
        self.th = []                 # 전체 이력 (kbps)
        self.by_vid = {}             # vid별 인덱스별 기록

    def append_throughput(self, event):
        vid = parse_vid(event.get('url', ''))
        if vid not in self.by_vid: self.by_vid[vid] = []
        idx = int(event.get('lastRequest', -1))
        val = event.get('bandwidthEst', None)
        if idx < 0 or val is None: return
        while len(self.by_vid[vid]) < idx:
            self.by_vid[vid].append(None)
        if idx == len(self.by_vid[vid]):
            self.by_vid[vid].append(val); self.th.append(val)
        elif self.by_vid[vid][idx] is None:
            self.by_vid[vid][idx] = val; self.th.append(val)

    def get(self, fallback=100.0):
        if not self.th: return fallback
        window = [v for v in self.th[-5:] if v and v > 0]
        if not window: return fallback
        rev = sum(1.0/v for v in window)
        return len(window)/rev

class DataLogger:
    def __init__(self, logfile):
        self.download_events = {}  # 메모리 디버그만
        try:
            from qoe_logger import QoELogger as _QoELogger
        except Exception:
            class _QoELogger:
                def on_session_end(self, vid, view_time): pass
                def on_download(self, event): pass
        self.qoe_logger = _QoELogger(logfile)

    def open(self):
        pass  # 파일 안 엶

    def close(self):
        pass

    def append_swipe(self, event):
        try:
            vid = parse_vid(event.get('url',''))
            self.qoe_logger.on_session_end(vid, float(event.get('viewTime') or 0.0))
        except Exception:
            pass

    def append_download(self, event):
        ev = copy.deepcopy(event)
        ev['vid'] = parse_vid(ev.get('url',''))
        try:
            self.qoe_logger.on_download(ev)
        except Exception:
            pass

# ======== 슬롯 매핑 규칙 ========
def choose_index(vid, pid, cpi, N, vid_idx):
    """
    1) vid가 이미 라칭되어 있으면 그 idx
    2) 없다면 cpi가 유효(>=0)면 (pid - cpi) % N, 아니면 pid % N
    """
    if vid in vid_idx: return vid_idx[vid]
    if isinstance(cpi, int) and cpi >= 0:
        idx = (int(pid) - int(cpi)) % N
    else:
        idx = int(pid) % N
    vid_idx[vid] = idx
    print(f"[VID IDX] vid={vid}, idx={idx}")
    return idx

# ======== 확률 없이 쓰는 MPC (로컬 완전탐색) ========
def mpc_no_prob(
    events,           # 준비된 이벤트 부분리스트 (url/chunksize_list/lastRequest/buffer 등)
    bitrate_profile,  # events와 1:1 매칭되는 [S][Q] (kbits)
    estimate_throughput_kbps,
    chunk_length=CHUNK_LENGTH,
    horizon=5,             # 지평 H (보통 3~5)
    rebuf_penalty=4.3,     # QoE: 리버퍼 패널티(표준 예시)
    smooth_penalty=1.0,    # QoE: 스무딩 패널티(표준 예시)
    util_mode='lin'        # 'lin' (선형 유틸) 또는 'log'
):
    """
    반환: 길이 len(events)의 리스트. 선택된 하나만 품질 인덱스, 나머지는 -2.
    """
    M = len(events)
    ret = [-2]*M
    if M == 0 or not bitrate_profile or estimate_throughput_kbps is None or estimate_throughput_kbps <= 0:
        return ret

    # 1) 각 이벤트의 상태 파싱
    total_segments = [len(bitrate_profile[i]) if i < len(bitrate_profile) else 0 for i in range(M)]
    buffer_len = []
    last_q = []
    for i, ev in enumerate(events):
        S = total_segments[i]
        last_req = int(ev.get('lastRequest', -1))
        buffer_len.append(clamp(last_req+1, 0, S))
        last_q.append(int(ev.get('lastquality', -1)))

    # 2) 어떤 비디오를 받을지 선택
    #    - 0번 스트림이 미완이면 0번 우선 (현재 재생)
    #    - 아니면 버퍼가 0인 애, 그래도 없으면 최소 버퍼
    def pick_video_idx():
        if buffer_len[0] < total_segments[0]:
            return 0
        sel, best = -1, 10**9
        # 버퍼 0 우선
        for i in range(1, M):
            if buffer_len[i] < total_segments[i] and buffer_len[i] == 0:
                return i
        # 최소 버퍼
        for i in range(1, M):
            if buffer_len[i] < total_segments[i] and buffer_len[i] < best:
                best = buffer_len[i]; sel = i
        return sel

    vi = pick_video_idx()
    if vi < 0: return ret  # 받을 게 없음

    next_seg_idx = buffer_len[vi]
    remain = max(0, total_segments[vi] - next_seg_idx)
    H = min(horizon, remain)
    if H <= 0: return ret

    Q = len(bitrate_profile[vi][0]) if total_segments[vi] > 0 else 0
    if Q == 0: return ret

    # 3) 시작 버퍼(초)
    #    - 0번(재생 중)은 플레이어가 알려준 'buffer'(초)를 그대로 사용 (가장 정확)
    #    - 그 외는 '받아둔 세그 * chunk_length' 근사
    if vi == 0:
        start_buffer_sec = float(events[0].get('buffer', 0.0) or 0.0)
    else:
        start_buffer_sec = buffer_len[vi] * float(chunk_length)

    prev_q = last_q[vi]
    sizes = [
        [bitrate_profile[vi][next_seg_idx + pos][q] for q in range(Q)]
        for pos in range(H)
    ]
    thr = float(estimate_throughput_kbps)

    best_reward, best_first_q = -1e30, 0
    for combo in itertools.product(range(Q), repeat=H):
        buf = start_buffer_sec
        rebuf = 0.0
        util_sum = 0.0
        smooth = 0.0
        lastq = prev_q

        for pos, q in enumerate(combo):
            size_kbits = sizes[pos][q]
            dl_time = size_kbits / max(thr, 1e-6)  # 초

            if buf < dl_time:
                rebuf += (dl_time - buf)
                buf = 0.0
            else:
                buf -= dl_time
            buf += float(chunk_length)

            # 유틸리티 (선형 또는 로그)
            br_kbps = size_kbits / float(chunk_length)
            if util_mode == 'log':
                base = max(sizes[pos][0] / float(chunk_length), 1e-6)
                util_sum += math.log(max(br_kbps, 1e-6) / base)
                if lastq >= 0:
                    prev_b = max(sizes[pos][lastq] / float(chunk_length), 1e-6)
                    smooth += abs(math.log(br_kbps) - math.log(prev_b))
            else:  # 'lin'
                util_sum += br_kbps
                if lastq >= 0:
                    prev_b = sizes[pos][lastq] / float(chunk_length)
                    smooth += abs(br_kbps - prev_b)

            lastq = q

        # QoE = 유틸(Mbps 스케일 맞춤) - 4.3*리버퍼 - 1.0*스무딩
        reward = (util_sum / 1000.0) - rebuf_penalty * rebuf - (smooth_penalty * smooth / 1000.0)

        # 동률이면 첫 q가 더 높은 쪽을 선호
        if reward > best_reward or (abs(reward - best_reward) < 1e-9 and combo[0] > best_first_q):
            best_reward, best_first_q = reward, combo[0]

    ret[vi] = int(best_first_q)
    print(f"[MPC-NO-PROB] choose vi={vi} next={next_seg_idx} buf={start_buffer_sec:.2f}s thr={thr:.0f}kbps -> q={ret[vi]}")
    return ret

# ======== HTTP 핸들러 ========
def make_handler(input_dict, logger, tp_estimator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _read_body(self):
            te = (self.headers.get('Transfer-Encoding') or '').lower()
            if 'chunked' in te:
                body = b''
                while True:
                    szline = self.rfile.readline()
                    if not szline: break
                    szline = szline.strip()
                    if not szline: continue
                    size = int(szline.split(b';', 1)[0], 16)
                    if size == 0:
                        # trailer skip
                        while True:
                            tr = self.rfile.readline()
                            if tr in (b'\r\n', b'\n', b''): break
                        break
                    body += self.rfile.read(size)
                    _ = self.rfile.read(2)  # \r\n
                return body
            length = int(self.headers.get('Content-Length', '0'))
            return self.rfile.read(length) if length > 0 else b''

        def _write_text(self, payload: bytes):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Requested-With')
            self.send_header('Access-Control-Max-Age', '86400')
            self.end_headers()

        def do_GET(self):
            payload = b"console.log('abr-server ok');"
            self.send_response(200)
            self.send_header('Cache-Control', 'max-age=60')
            self.send_header('Content-Type', 'application/javascript; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            raw = self._read_body()
            try:
                post = json.loads(raw.decode('utf-8') if raw else '{}')
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return

            # 종료 신호
            if post.get('isinfo') == -1:
                self._write_text(b"-2")
                try: sys.exit(0)
                except SystemExit: os._exit(0)
                return

            # 1) 정보 수집 단계
            if post.get('isinfo') == 1:
                input_dict['info_phase'] = True
                N = len(input_dict['events_record'])
                vid = parse_vid(post.get('url', ''))
                pid = int(post.get('playerId', 0))
                cpi = post.get('currentPlayerIdx', None)
                idx = choose_index(vid, pid, cpi, N, input_dict['vid_idx'])

                ev = copy.deepcopy(post)
                normalize_event_inplace(ev)
                ev["event_type"] = "video_info"   # 02.06
                logger.append_download(ev)  # 02.06
                input_dict['events_record'][idx] = ev

                # logger.append_download(post)
                tp_estimator.append_throughput(post)

                print(f"[MAP/INFO] pid={pid} cpi={cpi} -> idx={idx} vid={vid} lastRequest={post.get('lastRequest')}")
                self._write_text(b"-2")
                return
            
            if post.get("Type") == "qoe_finish":
                vid = post.get("vid") or parse_vid(post.get("url",""))
                wt  = float(post.get("watch_time") or 0.0)
                dur = float(post.get("duration") or 0.0)
                # Logging 02.06
                end_ev = {"url": post.get("url",""), "vid": vid, "viewTime": wt, "duration": dur}
                logger.append_swipe(end_ev)
                self._write_text(b"0")
                return

            # 3) 의사결정 단계 (info_phase가 True일 때 갱신)
            if input_dict['info_phase'] is True:
                # 준비된 이벤트만 추출
                ready_idx, ready_events = [], []
                for i, ev in enumerate(input_dict['events_record']):
                    if isinstance(ev, dict) and ev.get('url') and ev.get('chunksize_list'):
                        # print()
                        # print()
                        # print("ev.url : ", ev.get('url'))
                        # print("ev.chunksize_list : ", ev.get('chunksize_list'))
                        # print()
                        # print()
                        ready_idx.append(i); ready_events.append(ev)

                if not ready_events:
                    self._write_text(b"-2"); return

                # 비트레이트 프로파일(kbits)
                bitrate = get_bitrate_kbits_sq(ready_events)
                thr = tp_estimator.get()

                # 확률 없는 MPC
                ret_small = mpc_no_prob(ready_events, bitrate, thr)

                # 전체 슬롯으로 매핑
                plan = [-2] * len(input_dict['events_record'])
                for k, i in enumerate(ready_idx):
                    plan[i] = ret_small[k]
                input_dict['buffer_plan'] = plan
                input_dict['ITER_CNT'] += 1

            input_dict['info_phase'] = False

            # 4) 현재 요청에 대한 응답 코드 선택
            planN = len(input_dict['buffer_plan'])
            if planN == 0:
                self._write_text(b"-2"); return

            vid = parse_vid(post.get('url', ''))
            pid = int(post.get('playerId', 0))
            cpi = post.get('currentPlayerIdx', None)
            idx = choose_index(vid, pid, cpi, planN, input_dict['vid_idx'])
            if idx < 0 or idx >= planN:
                self._write_text(b"-2"); return

            return_code = input_dict['buffer_plan'][idx]
            print(f"[MAP/DONE] pid={pid} cpi={cpi} -> idx={idx} vid={vid} "
                  f"chunkId={post.get('chunkId')} lastRequest={post.get('lastRequest')} return_code={return_code}\n")

            if return_code != -2:
                print(f" ::: {int(parse_vid(post.get('url')))} ::: lastRequest: {post.get('lastRequest')}, "
                      f"currentPlayerIdx: {post.get('currentPlayerIdx')}, playerId: {post.get('playerId')}, "
                      f"lastquality: {int(post.get('lastquality', -1))}")

            self._write_text(str(return_code).encode('utf-8'))

        def log_message(self, fmt, *args):  # 콘솔 잡음 제거
            return

    return Handler

# ======== 서버 실행 ========
def run(port=8334, log_file_path=LOG_FILE):
    logger = DataLogger(log_file_path); logger.open()
    tp_est = ThroughputEstimator()

    state = {
        'vid_idx': {},                 # vid -> 슬롯 인덱스 라칭
        'info_phase': True,
        'ITER_CNT': 0,
        'buffer_plan': [-2 for _ in range(5)],
        'events_record': [{} for _ in range(5)]
    }

    handler = make_handler(state, logger, tp_est)
    srv = ThreadingHTTPServer(('0.0.0.0', port), handler)
    print(f"Listening on port {port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default="exp")
    ap.add_argument('--port', type=int, default=8334)
    args = ap.parse_args()
    log_dir = f"/home/harim/Dashlet_for_www/data/{args.log}"
    os.makedirs(log_dir, exist_ok=True)
    logfile = f"{log_dir}/qoe_out.log"
    run(port=args.port, log_file_path=logfile)

if __name__ == "__main__":
    main()
