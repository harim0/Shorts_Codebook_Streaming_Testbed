# GUIDE: Codebook-Switch SR Streaming Testbed

> **대상**: 연구팀 내부. ML/스트리밍 도메인 지식 전제. 코드 레벨 세부사항 포함.

---

## Part 1. 배경 및 목표

### 1.1 Codebook Switching SR

tex-VQVAE 기반 모델로, HR 프레임의 texture 정보를 codebook index (i-matrix)로 압축해 전송하고, 클라이언트에서 LR + i-matrix → Decoder → 1080p SR 복원하는 구조.

**핵심 전송 단위 (클러스터당 1쌍)**:
- `imatrix.pt` : HR → Encoder → L2-norm → argmin → `(n_frames, 240, 135)` int16 (~9.3MB)
- `decoder.pt` : tex-VQVAE Decoder 가중치 (~880KB)
- `coarse_sr.pt` : SRResNet Coarse-SR 가중치 (~929KB)
- `segment_0_{idx}.m4s` : 270p LR 세그먼트 (~183KB/38KB)

**Portrait 해상도 (Shorts 기준 9:16)**:

| 단계 | Width | Height |
|------|:-----:|:------:|
| LR (전송) | 270 | 480 |
| HR (복원) | 1080 | 1920 |

### 1.2 i-matrix 생성 원리

```python
# tex_vqvae8.py forward() 참조:
# z = self.encoder(hr_curr)   ← HR(1,3,1920,1080) 직접 입력
# precompute_imatrix.py:
z = model.encoder(hr_t)              # (1, embed_dim, 240, 135)
z_low = model.quantize.proj_down(z)  # (1, codebook_dim, 240, 135)
z_low_norm = F.normalize(z_low.reshape(-1, C), p=2, dim=-1)
indices = cdist(z_low_norm, e_norm).argmin(dim=-1)  # (240×135,)
# → shape (n_frames, 240, 135), dtype int16
```

**주의**: LR → coarse_sr → encoder 경로는 inference-time 구조체 복원용이고, i-matrix는 반드시 HR → encoder로 추출해야 함.

---

## Part 2. 시스템 아키텍처

### 전체 요청-응답 흐름

```
Browser (dash.all.debug.js)
│
├─ [MPD 로드 시, 1회]
│   ├─ GET  CDN:8080 /dash/data/{class}/{vid}/imatrix.pt
│   │   └─ POST Flask:8081 /imatrix  → dnn_queue('imatrix', vid, path)
│   │
│   └─ GET  CDN:8080 /dash/model/{class}/decoder.pt
│      GET  CDN:8080 /dash/model/{class}/coarse_sr.pt
│          └─ POST Flask:8081 /dnn  → dnn_queue('dnn_model', class, dec, csr)
│                                     (decoder+coarse_sr 둘 다 도착 시)
│
└─ [세그먼트 루프]
    ├─ POST ABR:8334  (buffer, throughput, chunksize_list → quality 0-3)
    ├─ GET  CDN:8080  segment_0_{idx}.m4s  (270p LR)
    └─ POST Flask:8081 /uploader  (segment bytes, quality, fps, vid)
                └─ SR pipeline → 1080p mp4 반환 → MSE buffer append
```

### SR 멀티프로세스 파이프라인 (process.py)

```
Flask /uploader
  └─ decode_queue.put((init_m4s, media_m4s, pipe, video_info))

decode_process:
  header + media m4s 병합 → input.mp4
  data_queue.put('configure', targetScale=4, targetWidth=270, vid, class)
  OpenCV VideoCapture → cv2.resize((270, 480)) → ByteTensor(H=480, W=270, C=3)
  shared_tensor_list[270][frame_idx % 120].copy_(input_t)   ← width-key, portrait
  data_queue.put('frame', frame_count)

sr_process (super_resolution_threading):
  ├─ [load_dnn_chunk thread]  dnn_queue 메시지 처리:
  │   ├─ 'dnn_model'  → model.load_model(decoder.pt, coarse_sr.pt)
  │   └─ 'imatrix'    → IMATRIX_STORE[vid] = torch.load(path)
  │
  └─ [process_video_chunk thread]  data_queue 메시지 처리:
      ├─ 'configure'  → targetWidth=270, vid/class 갱신
      ├─ 'process_dir'→ process_dir 갱신
      └─ 'frame'      →
          lr_prev = shared_tensor_list[270][prev_idx]   # (480, 270, 3)
          lr_curr = shared_tensor_list[270][curr_idx]   # (480, 270, 3)
          imatrix = IMATRIX_STORE[vid]                  # (n_frames, 240, 135)
          model.set_imatrix(imatrix)
          output_ = model.infer_with_imatrix(lr_prev, lr_curr, frame_idx)
          shared_tensor_list[1080][curr_idx].copy_(output_)  # (1920, 1080, 3)
          encode_queue.put('frame', curr_idx)

encode_process:
  ffmpeg: rawvideo 1080x1920 rgb24 → h264 libx264 ultrafast → output.mp4
  output_input.send(('output', output_mp4_path, infer_idx))
  → Flask /uploader: send_file(swap_file) → arraybuffer 응답
```

**shared_tensor_list 구조**:
```python
res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
# key=width, shape=(H, W, C) portrait
shared_tensor_list[270]  → list of ByteTensor(480, 270, 3)   # LR 버퍼
shared_tensor_list[1080] → list of ByteTensor(1920, 1080, 3) # SR 출력 버퍼
SHARED_QUEUE_LEN = 120   # 30fps × 4sec
```

---

## Part 3. 컴포넌트 상세

### 3.1 dash.all.debug.js — imatrix + DNN 요청

**imatrix 다운로드 트리거** (ManifestLoader, MPD 로드 직후):

```javascript
// line ~25828
var imatrixUrl = cdnBase + '/dash/data/' + vidId + '/imatrix.pt';
xhr.open('GET', imatrixUrl, true);
// → 수신 후:
fd.append('imatrix', blob);
fd.append('jsondata', JSON.stringify({ vid: vidId }));  // "Animal/SDR_Animal_23rp"
xhr2.open('POST', 'http://HOST:8081/imatrix', true);
```

**decoder.pt + coarse_sr.pt 다운로드** (send_DNN_Request_codebook, ScheduleController):

```javascript
// line ~37988
var className = ctx.vid.split('/')[0];
var baseUrl = "http://163.152.162.202:8080/dash/model/" + className + "/";
files = ["decoder.pt", "coarse_sr.pt"];
// GET CDN → blob → POST /dnn {file_name, class}
// 둘 다 완료 시: ctx.reqType = "video" → 세그먼트 다운 시작
```

**세그먼트 → SR 송신** (send2DNNprocess, FragmentController):

```javascript
// line ~35831
xhr.open("POST", "http://HOST:8081/uploader", true);
xhr.responseType = "arraybuffer";
data.append('videofile', blob_of_segment);
data.append('jsondata', JSON.stringify(chunk));  // quality, index, fps, duration, vid
// 응답: SR된 1080p mp4 arraybuffer → MSE buffer append
```

### 3.2 dnn_appLocalServer.py — Flask Routes

| Route | Method | 역할 | Queue |
|-------|--------|------|-------|
| `/imatrix` | POST | imatrix.pt 저장 → dnn_queue | `dnn_queue.put(('imatrix', vid, path))` |
| `/dnn` | POST | decoder.pt / coarse_sr.pt 저장; 둘 다 있으면 → dnn_queue | `dnn_queue.put(('dnn_model', class, dec, csr))` |
| `/uploader` | POST | 세그먼트 수신 → decode_queue → SR 완료 대기 → 1080p mp4 응답 | `decode_queue.put(...)` |
| `/dnn_config` | POST | 초기 handshake (legacy, inference_idx 테스트용) | `dnn_queue.put(('test_dnn', ...))` |
| `/uploadPlayback` | GET | 재생 메트릭 수집 | — |
| `/uploadRebuffer` | GET | 리버퍼링 이벤트 수집 | — |

**Flask `threaded=True`** 필수: `/uploader`의 블로킹 recv() 중에도 `/dnn`, `/imatrix` 동시 처리를 위해.

### 3.3 dnn_custom_server_mpc.py — ABR (:8334)

MPC 5-horizon 탐색으로 quality 결정:

- `isinfo=1` 요청: 브라우저가 buffer/throughput/chunksize 상태 보고 → 슬롯 기록
- `isinfo=0` 요청: MPC 탐색 → quality 0–3 반환
- `dnn_mode=1` 일 때: buffer ≥ DNN_BUFFER_TH 이면 `return_code=Q` (DNN 파일 다운로드 지시)
- QUEUE_LEN=5 슬롯: pid 기반 슬롯 라칭 (`vid_idx` 딕셔너리)

---

## Part 4. CDN 서버 구조

```
cdn-server/contentServer/dash/
├── data/{class}/{vid}/
│   ├── multi_resolution.mpd       # DASH manifest (4 Representation: 270p/360p/540p/1080p)
│   ├── imatrix.pt                 # (n_frames, 240, 135) int16, ~9.3MB
│   ├── init_0.m4s                 # 270p 초기화 세그먼트
│   └── segment_0_{1..N}.m4s      # 270p LR 세그먼트 (~183KB)
└── model/{class}/
    ├── decoder.pt                 # ~880KB
    └── coarse_sr.pt               # ~929KB
```

**i-matrix 사전 생성**:
```bash
cd cdn-server/contentServer/dash
python precompute_imatrix.py --overwrite
# HR mp4(1080×1920, 150fr) → encoder → (150, 240, 135) int16 → imatrix.pt
```

---

## Part 5. 시뮬레이션 결과

30개 영상(5 클래스), 120–150 프레임 기준:

| 메트릭 | 평균 | 표준편차 |
|--------|:----:|:-------:|
| PSNR E2E (Codebook SR) | **31.44 dB** | 4.39 |
| PSNR Coarse-SR 베이스라인 | 29.80 dB | 4.19 |
| **PSNR 향상** | **+1.64 dB** | — |
| SR 속도 | 9.3 fps | 0.14 |
| 전체 처리 (decode+SR+encode) | 18.0 s/clip | 0.31 |

전송 크기 (270p, 4sec 세그먼트 기준):
- i-matrix: ~9.3MB, decoder: ~880KB, coarse_sr: ~929KB, seg1: ~183KB, seg2: ~38KB

---

## Part 6. 실행 방법

```bash
# 1. i-matrix 사전 생성 (최초 1회 또는 모델 변경 시)
cd cdn-server/contentServer/dash
python precompute_imatrix.py --overwrite

# 2. CDN
sudo systemctl start nginx

# 3. ABR Server
cd abr-server
python dnn_custom_server_mpc.py --port 8334

# 4. Flask + SR Server
cd web-server
PYTHONUNBUFFERED=1 python dnn_appLocalServer.py

# 5. 시뮬레이션 (실제 브라우저 없이)
cd simulation
python run_simulation.py
python plot_results.py
```

---

## Part 7. ToDo — Mobile Testbed

NEMO (MobiCom'20) 참조, Android 클라이언트 구현 예정.

| 항목 | 현재 (서버 GPU 추론) | 목표 (Android 기기 추론) |
|------|:---:|:---:|
| 추론 위치 | process.py (서버) | PyTorch Mobile / ExecuTorch (기기) |
| 전송 구조 | 270p → 서버 SR → 1080p 반환 | 270p + imatrix → 기기 SR |
| Codebook | 서버 메모리 | 앱 번들 사전 배포 |
| 플레이어 | Browser (DASH.js) | Android ExoPlayer |
| 추론 엔진 | PyTorch CUDA | INT8, XNNPACK backend |

```bash
# Mahimahi LTE 에뮬레이션 예시
mm-link traces/ATT-LTE-driving.up traces/ATT-LTE-driving.down \
  mm-delay 25 \
  -- python web-server/dnn_appLocalServer.py
```

---

## References

```
[1] H. Yeo et al., "Neural Adaptive Content-aware Internet Video Delivery," OSDI 2018.
[2] Z. Li et al., "Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming," NSDI 2023.
[3] H. Yeo et al., "NEMO: Enabling Neural-enhanced Video Streaming on Commodity Mobile Devices," MobiCom 2020.
[4] H. Mao et al., "Neural Adaptive Video Streaming with Pensieve," SIGCOMM 2017.
[5] R. Netravali et al., "Mahimahi: Accurate Record-and-Replay for HTTP," ATC 2015.
```
