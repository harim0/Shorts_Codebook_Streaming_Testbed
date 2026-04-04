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


### Nginx 설정 파일 구조

시스템 nginx (`/etc/nginx/`) 사용. 두 개의 설정 파일이 `sites-available/` 에 있고 `sites-enabled/`에 symlink로 활성화되어 있어야 한다.

```bash
# 활성화 확인
ls -la /etc/nginx/sites-enabled/
# dashlet-cdn -> /etc/nginx/sites-available/dashlet-cdn
# abr-8333.conf -> /etc/nginx/sites-available/abr-8333.conf

# 없으면 symlink 생성
sudo ln -s /etc/nginx/sites-available/dashlet-cdn /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/abr-8333.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

#### `dashlet-cdn` (포트 8080) — CDN 서버 **[필수]**

| Location | 역할 |
|----------|------|
| `/dash/data/{vid}/` | MPD 매니페스트, 해상도별 `.m4s` segment 서빙 |
| `/dash/model/{vid}/` | DNN 모델 chunk (`DNN_chunk_N.pth`) 서빙 |

이 파일이 없거나 비활성화되면 브라우저가 영상/MPD를 전혀 읽지 못한다.

```bash
harim@server3:~$ cat /etc/nginx/sites-available/abr-8333.conf 
map $http_origin $cors_allow_origin {
    default "";
    "~^http://163\.152\.162\.202:8081$" $http_origin;
    "~^http://localhost:9989$"         $http_origin;
}

server {
    listen 8333;
    server_name 163.152.162.202;

    # (선택) 업스트림(Python)이 보낸 CORS 헤더를 숨겨 중복 방지
    proxy_hide_header Access-Control-Allow-Origin;
    proxy_hide_header Access-Control-Allow-Methods;
    proxy_hide_header Access-Control-Allow-Headers;
    proxy_hide_header Access-Control-Allow-Credentials;
    proxy_hide_header Vary;
        location / {
        if ($request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin $cors_allow_origin always;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
            add_header Access-Control-Allow-Headers $http_access_control_request_headers always;
            add_header Access-Control-Max-Age 86400 always;
            return 204;
        }

        proxy_pass http://localhost:8334;
        proxy_http_version 1.1;
        proxy_set_header Host $host;

        add_header Access-Control-Allow-Origin $cors_allow_origin always;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
        add_header Access-Control-Allow-Headers $http_access_control_request_headers always;
        add_header Access-Control-Allow-Credentials "true" always;

        proxy_set_header X-Real-IP       $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### `abr-8333.conf` (포트 8333 → 8334) — ABR 역방향 프록시 **[필수]**

`dash.js`는 ABR 결정을 위해 `window.location.hostname:8333`에 POST를 보낸다.
Python ABR 서버(`dnn_custom_server_mpc.py`)는 포트 **8334**에서 동작하며 자체적으로 `Access-Control-Allow-Origin: *`를 붙이지만,
nginx 8333 레이어가 이를 숨기고 허용 origin을 `http://{SERVER_IP}:8081`로 제한한다.

> **이 파일 없이 구동하려면**: `dash.all.debug.js`에서 8333 → 8334로 포트 번호 변경 후 nginx 없이 Python 서버 CORS(`*`)를 그대로 사용 가능.

```bash
# /etc/nginx/sites-available/dashlet-cdn
server {
    listen 8080 default_server;
    server_name _;

    root /home/harim/Dashlet_www;
    index index.html index.htm;

    location ^~ /dash/data/ {
        alias /home/harim/Dashlet_www/cdn-server/contentServer/dash/data/;
        autoindex on;
        etag off;                     # ETag 생성 X
        if_modified_since off;        # If-Modified-Since 무시
        
         add_header Cache-Control "no-store" always;
        add_header Access-Control-Allow-Origin http://163.152.162.202:8081 always;
        add_header Access-Control-Allow-Headers * always;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;

        types {
            application/dash+xml mpd;
            video/iso.segment m4s;
            video/mp4 mp4 m4v;
        }
        default_type application/octet-stream;
        try_files $uri =404;
    } 

    location ^~ /dash/model/ {

    # OPTIONS preflight 전용 location
    if ($request_method = OPTIONS) {
        return 204;
    }

    # GET/POST 응답
    add_header Access-Control-Allow-Origin "http://163.152.162.202:8081" always;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, Content-Type, X-Requested-With" always;
    add_header Access-Control-Expose-Headers "Content-Range, Accept-Ranges" always;

    alias /home/harim/Dashlet_www/cdn-server/contentServer/dash/model/;
    autoindex on;
    etag off;
    if_modified_since off;
    add_header Accept-Ranges bytes;

    default_type application/octet-stream;
    try_files $uri =404;
        }
 
}
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
PYTHONUNBUFFERED=1 python dnn_appLocalServer.py --quality scale4

# 5. 시뮬레이션 (실제 브라우저 없이)
cd simulation
python run_simulation.py
python plot_results.py
```

---
<!-- 
## Part 7. ToDo — Mobile Testbed (NEMO 방식)

### 현재 테스트베드의 구조적 한계

현재 testbed는 **NAS_demo 구조를 그대로 상속**: Chrome Browser가 ABR/플레이어 역할을 하고, LR 세그먼트를 서버(Flask)로 전송하면 서버 GPU에서 SR을 수행한 뒤 1080p를 돌려주는 구조다.

```
[현재 testbed]
Android/Browser
  → LR segment → POST /uploader → 서버 GPU (process.py SR) → 1080p 반환
  (SR이 서버에서 일어남 = 실제 배포 구조가 아님, 파이프라인 검증용)
```

이는 **실제 Deploy 목표와 다르다**. 실제 목표는 클라이언트가 LR + i-matrix + model을 받아 기기 자체에서 SR 추론을 수행하는 것.

### 배포 목표 구조 (NEMO-style)

NEMO (MobiCom'20) 방식: SR을 비디오 코덱 디코더 내부에 통합, Android 기기에서 직접 추론.

```
[배포 목표]
CDN → LR 270p segment + imatrix.pt + decoder.pt + coarse_sr.pt 전송
Android 기기:
  codec decoder (libvpx / ExoPlayer MediaCodec)
    └─ anchor frame마다 SR DNN 호출 (Codebook SR: imatrix + LR → 1080p)
    └─ non-anchor frame: 보간 (interpolation)
  Codebook(quantize.embedding.weight) → 앱 번들로 사전 배포 (per-cluster)
```

| 항목 | 현재 테스트베드 (검증용) | 배포 목표 (NEMO-style) |
|------|------|------|
| 클라이언트 | Chrome Browser (NAS_demo 상속) | Android ExoPlayer |
| SR 위치 | 서버 (Flask process.py, 검증 편의) | Android 기기 내 (codec 통합) |
| 전송 구조 | LR → 서버 SR → 1080p 반환 | LR + imatrix + model → 기기 SR |
| Codebook | 서버 메모리 로드 | 앱 번들 사전 배포 (per-cluster) |
| 추론 엔진 | PyTorch CUDA (서버 GPU) | PyTorch Mobile / SNPE (Android NPU) |
| 코덱 통합 | 없음 (별도 파이프라인) | libvpx 수정 또는 MediaCodec 후처리 |

### NEMO 핵심 기법 (적용 예정)

- **Anchor frame 선택**: GOP(Group of Pictures) 기반으로 SR 적용 프레임 선택 → 나머지는 보간
- **codec 통합**: libvpx decoder 내 DNN 호출 삽입 (NEMO 참조) 또는 ExoPlayer MediaCodec 후처리
- **모델 경량화**: INT8 quantization, XNNPACK backend (PyTorch Mobile)
- **전송 절감**: 270p(LR)만 전송 → 기기에서 x4 복원 → 1080p 출력

```bash
# Mahimahi LTE 에뮬레이션 예시 (testbed 대역폭 제한)
mm-link traces/ATT-LTE-driving.up traces/ATT-LTE-driving.down \
  mm-delay 25 \
  -- python web-server/dnn_appLocalServer.py
```

--- -->

## References

```
[1] H. Yeo et al., "Neural Adaptive Content-aware Internet Video Delivery," OSDI 2018.
[2] Z. Li et al., "Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming," NSDI 2023.
[3] H. Yeo et al., "NEMO: Enabling Neural-enhanced Video Streaming on Commodity Mobile Devices," MobiCom 2020.
[4] H. Mao et al., "Neural Adaptive Video Streaming with Pensieve," SIGCOMM 2017.
[5] R. Netravali et al., "Mahimahi: Accurate Record-and-Replay for HTTP," ATC 2015.
```
