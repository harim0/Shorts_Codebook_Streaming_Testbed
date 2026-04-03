# Shorts Codebook-Switch SR Streaming Testbed

> Shorts(TikTok-style) 환경에서 **Codebook Switching SR** (tex-VQVAE + SRResNet) 모델을 DASH ABR 파이프라인에 통합한 실험 테스트베드.
> 270p(LR) 세그먼트만 전송하고 클라이언트에서 x4 SR로 1080p를 복원하는 구조.

---

## 모델 개요

| 단계 | 내용 |
|------|------|
| **Stage 1** | tex-VQVAE (Encoder + Decoder + Codebook + Coarse-SR(SRResNet)) 일반 학습 |
| **Stage 2** | codebook histogram 기반 KNN 클러스터링 + Meta-learning (REPTILE) → per-cluster 특화 모델 |
| **전송** | LR 270p 세그먼트 + i-matrix + Decoder + Coarse-SR (클러스터당 1쌍) |
| **클라이언트** | 사전 배포된 Codebook(quantize.embedding.weight) + 수신 모델로 실시간 SR |

**Portrait 기준** (Shorts/TikTok): LR W=270, H=480 → HR W=1080, H=1920 (x4 복원, 9:16)

---

## 시스템 구성

| 컴포넌트 | 역할 | 포트 | 핵심 파일 |
|---------|------|------|---------|
| **Browser App** | sequence 재생, prefetch 큐(QUEUE_LEN=5) 관리 | — | `web-server/static/js/player.js` |
| **dash.js (수정)** | MPD 파싱, imatrix/model DL, segment SR 송신 | — | `web-server/static/js/dash.all.debug.js` |
| **Flask Server** | 세그먼트·모델·imatrix 수신 → SR 파이프라인 제어 | **:8081** | `web-server/dnn_appLocalServer.py` |
| **ABR Server** | MPC: buffer+throughput → quality 결정 | **:8334** | `abr-server/dnn_custom_server_mpc.py` |
| **CDN Server** | Nginx: MPD, 270p segment, imatrix, 모델 파일 제공 | **:8080** | `cdn-server/contentServer/` |
| **SR Pipeline** | decode → CodebookSR 추론 → encode | — | `web-server/super_resolution/process.py` |

---

## 데이터 플로우

```
Browser
  → GET  CDN:8080  /dash/data/{class}/{vid}/multi_resolution.mpd
  → GET  CDN:8080  /dash/data/{class}/{vid}/imatrix.pt
  → POST Flask:8081 /imatrix   (i-matrix → dnn_queue)
  → GET  CDN:8080  /dash/model/{class}/decoder.pt
  → GET  CDN:8080  /dash/model/{class}/coarse_sr.pt
  → POST Flask:8081 /dnn       (model → dnn_queue, 클래스당 1회)
  ─── 세그먼트 루프 ───
  → POST ABR:8334              (buffer/throughput → quality 결정)
  → GET  CDN:8080  /dash/data/{class}/{vid}/segment_0_{idx}.m4s  (270p LR)
  → POST Flask:8081 /uploader  (segment bytes → SR pipeline → 1080p mp4 반환)
  → MSE buffer append
```

---

## Quick Start

```bash
# 1. CDN (Nginx)
sudo systemctl start nginx

# 2. ABR Server
cd abr-server
python dnn_custom_server_mpc.py

# 3. Flask + SR Server
cd web-server
python dnn_appLocalServer.py

# 4. 브라우저 (Chrome, DevTools → Disable cache)
# http://163.152.162.202:8081/
```

> 상세 설계, 컴포넌트 설명, 실험 결과 → [GUIDE.md](./GUIDE.md)

---

## 디렉토리 구조

```
Shorts_Codebook_Streaming_Testbed/
├── web-server/
│   ├── dnn_appLocalServer.py          # Flask 메인 서버 (SR + 라우팅)
│   └── super_resolution/
│       ├── process.py                 # 멀티프로세스 SR 파이프라인
│       ├── codebook_sr.py             # CodebookSR 추론 래퍼
│       └── model/
│           ├── tex_vqvae8.py          # tex-VQVAE 모델 정의
│           └── {class}/
│               ├── decoder.pt         # per-cluster Decoder 가중치
│               └── coarse_sr.pt       # per-cluster Coarse-SR 가중치
├── abr-server/
│   └── dnn_custom_server_mpc.py       # MPC ABR 서버 (:8334)
├── cdn-server/contentServer/
│   └── dash/
│       ├── data/{class}/{vid}/
│       │   ├── multi_resolution.mpd   # DASH manifest
│       │   ├── imatrix.pt             # HR→encoder 기반 i-matrix (150fr, 240×135)
│       │   └── segment_0_{idx}.m4s    # 270p LR 세그먼트
│       └── model/{class}/
│           ├── decoder.pt             # Decoder 가중치
│           └── coarse_sr.pt           # Coarse-SR 가중치
└── simulation/
    ├── run_simulation.py              # E2E 시뮬레이션 (PSNR 포함)
    ├── plot_results.py                # 결과 시각화
    └── results/sim_results.json       # 시뮬레이션 결과 (30 videos)
```

---

## ToDo: Mobile Testbed

NEMO (MobiCom'20) 참조 — Android ExoPlayer + PyTorch Mobile / ExecuTorch 기반 SR 추론으로 확장 예정.

| 항목 | 현재 (Browser) | 목표 (Android) |
|------|:---:|:---:|
| 추론 위치 | process.py (서버 GPU) | 기기 내 (PyTorch Mobile) |
| 전송 가정 | 서버에서 SR 후 1080p 반환 | 기기에서 직접 SR (270p만 수신) |
| Codebook | 서버 메모리 | 앱 번들 사전 배포 |

---

## References

```
[1] H. Yeo et al., "Neural Adaptive Content-aware Internet Video Delivery," OSDI 2018.
[2] Z. Li et al., "Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming," NSDI 2023.
[3] H. Yeo et al., "NEMO: Enabling Neural-enhanced Video Streaming on Commodity Mobile Devices," MobiCom 2020.
[4] H. Mao et al., "Neural Adaptive Video Streaming with Pensieve," SIGCOMM 2017.
```
