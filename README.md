# Shorts-Optimized Neural Adaptive Streaming Testbed

> Shorts(TikTok-style) 환경에서 **DNN Super-Resolution + ABR**을 통합 실험하기 위한 Streaming Testbed.
> [Dashlet (NSDI'23)](https://www.usenix.org/conference/nsdi23/presentation/li-zhuqi)의 Shorts prefetch 전략과 [NAS (OSDI'18)](https://www.usenix.org/conference/osdi18/presentation/yeo)의 scalable SR DNN 파이프라인을 결합.
> **Codebook Switching SR** (tex-VQVAE + SRResNet) 모델의 모바일 클라이언트 배포를 목표로 한다.

---

## Origin & Contribution

| Base Repo | Paper | 역할 | 본 repo 변경 |
|-----------|-------|------|------------|
| [PrincetonUniversity/Dashlet](https://github.com/PrincetonUniversity/Dashlet) | NSDI'23 | Shorts prefetch 큐, sequence 기반 재생 | `player.js`, `sequence.json` 확장 |
| [kaist-ina/NAS_demo](https://github.com/kaist-ina/NAS_demo) | OSDI'18 | SR DNN pipeline, ABR 연동 | `dnn_appLocalServer.py`, `process.py`, `dash.js` 수정 |

---

## System At a Glance

| Component | Role | Port | Key File |
|-----------|------|------|----------|
| **Browser App** | sequence 재생, prefetch 큐 관리 | — | `web-server_www/static/js/player.js` |
| **dash.js (수정)** | MPD 파싱, ABR 연동, DNN 요청 | — | `web-server_www/static/js/dash.all.debug.js` |
| **App/Experiment Server** | Flask: DNN config/chunk 수신, SR 파이프라인 | **:8081** | `web-server_www/dnn_appLocalServer.py` |
| **ABR Server** | E-MPC: buffer+throughput → quality 결정 | **:8333/8334** | `abr-server/empc_server_dashlet.py` |
| **CDN Server** | Nginx: MPD, 영상 segment, DNN chunk 제공 | **:8080** | `cdn-server/contentServer/` |
| **SR Pipeline** | decode → NAS SR → encode (multi-process) | — | `web-server_www/super_resolution/process.py` |

---

## One-line Data Flow

```
Browser
  → GET MPD (CDN :8080)
  → POST state (ABR :8333) → quality decision
  → GET segment (CDN :8080)
  → POST /uploader (App :8081) → SR pipeline → MSE buffer
  [optional] → GET DNN_chunk (CDN) → POST /dnn_chunk → inference_idx 업데이트
```

---

## Quick Start

```bash
# 1. CDN (Nginx)
sudo systemctl start nginx

# 2. ABR Server
cd abr-server
python dnn_custom_server_mpc.py 

# 3. App/Experiment Server (SR + Flask)
cd web-server_www
python dnn_appLocalServer.py --quality ultra 

# 4. sequence.json 에 재생할 video pid 목록 설정
# web-server_www/static/sequence.json

# 5. 브라우저 (Chrome)
# DevTools → Network → Disable cache
# http://163.152.162.202:8081/
```

> 상세 설치 및 설정은 [GUIDE.md](./GUIDE.md) 참조.

---

## Directory Structure

```
Dashlet_www/
├── web-server_www/                 # App/Experiment Server
│   ├── dnn_appLocalServer.py       # Flask 메인 서버 (SR + DNN 라우팅)
│   ├── app-local.py                # SR 없는 경량 서버
│   ├── super_resolution/
│   │   ├── process.py              # 멀티프로세스 SR 파이프라인
│   │   └── model/
│   │       ├── NAS.py              # Scalable DNN (Multi_Network)
│   │       └── {low,medium,high,ultra}/   # pre-trained weights
│   ├── static/js/
│   │   ├── player.js               # SuperPlayer (sequence 재생 로직)
│   │   └── dash.all.debug.js       # 수정된 dash.js (DNN 연동)
│   └── static/sequence.json        # 재생 플레이리스트
├── abr-server/
│   ├── empc_server_dashlet.py      # E-MPC ABR server
│   └── abrAlgorithmCollection*.py  # ABR 알고리즘 모음
└── cdn-server/contentServer/
    ├── dash/data/{vid}/            # 해상도별 DASH segments
    └── dash/model/{vid}/ultra/     # DNN_chunk_*.pth
```

---

## ToDO : Mobile Testbed 구성 

### NEMO (MobiCom'20) 참조 구조

[kaist-ina/nemo](https://github.com/kaist-ina/nemo) — NAS 저자 동일 그룹, Android 모바일 SR 스트리밍 테스트베드.

| 항목 | NEMO / Palantir | 현재 (NAS+Dashlet/Codebook) |
|------|:---------------:|:------------------:|
| SR 위치 | 코덱 디코더 내 (libvpx 수정) | process.py 별도 파이프라인 |
| 추론 엔진 | Qualcomm SNPE SDK | PyTorch Mobile / ExecuTorch |
| 프레임 선택 | Anchor Point (GOP 기반) | inference_idx (레이어 기반) |
| 클라이언트 | Android ExoPlayer | Browser (DASH.js) → Android 확장 예정 |

### 필요 도구 (모바일 클라이언트 가정 시)

| 도구 | 역할 | 비고 |
|------|------|------|
| **Android Studio** | ExoPlayer 기반 플레이어 빌드 | NEMO `player/` 참조 |
| **Qualcomm SNPE SDK v1.40+** | Snapdragon NPU 추론 | TF → `.dlc` 변환 필요 |
| **PyTorch Mobile / ExecuTorch** | 범용 Android CPU 추론 | INT8, XNNPACK backend |
| **Android ADB** | 기기 연동 + 로그 수집 | Mahimahi UsbShell 연동 가능 |
| **ARM64 cross-compiler** | libvpx ARM64 빌드 | `nemo_client_arm64.sh` 참조 |
| **Mahimahi** | 네트워크 에뮬레이션 (LTE/5G 트레이스) | LinkShell + DelayShell 조합 |
| **Xvfb** | 헤드리스 브라우저 실행 | `sudo apt-get install xvfb` |
| **tc (traffic control)** | 간단한 대역폭 제한 | `tc qdisc add dev eth0 root tbf ...` |

```bash
# Mahimahi LTE 환경 예시 (2Mbps, 50ms RTT)
mm-link traces/ATT-LTE-driving.up traces/ATT-LTE-driving.down \
  mm-delay 25 \
  -- python web-server_www/dnn_appLocalServer.py
```


---

## References

```
[1] H. Yeo et al., "Neural Adaptive Content-aware Internet Video Delivery," USENIX OSDI 2018.
[2] Z. Li et al., "Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming," USENIX NSDI 2023.
[3] B. Guo et al., "LAR-SR: A Local Autoregressive Model for Image Super-Resolution," CVPR 2022.
[4] H. Yeo et al., "NEMO: Enabling Neural-enhanced Video Streaming on Commodity Mobile Devices," ACM MobiCom 2020.
[5] "Palantir: Efficient Super Resolution for Ultra-high-definition Live Streaming," ACM MMSys 2025.
[6] S. Park et al., "EOS: Energy-Optimized SR on Mobile Devices for Live 360-Degree Videos," ACM MobiCom 2025.
[7] H. Mao et al., "Neural Adaptive Video Streaming with Pensieve," ACM SIGCOMM 2017.
[8] R. Netravali et al., "Mahimahi: Accurate Record-and-Replay for HTTP," USENIX ATC 2015.
```

> 상세 설계, 컴포넌트 설명, 모바일 testbed, Codebook Switching 통합 계획 → **[GUIDE.md](./GUIDE.md)**
