# Mobile Testbed Plan — Codebook-Switch SR

## 컴포넌트 매핑 (Browser/Server → Android)

| 현재 (Browser/Server) | Android 대응 파일 |
|------|------|
| `player.js` — sequence 순회, prefetch, playPtr | `MainActivity.kt` — loadVideo(), advanceToNext() |
| `dash.all.debug.js` — SuperPlayer, ManifestLoader, DASH ABR | `ExoPlayer` + `Media3 DASH` (MediaItem.fromUri(mpd)) |
| `dash.all.debug.js` — triggerImatrixDownload() | `SideAssetManager.fetchImatrix()` |
| `dash.all.debug.js` — send_DNN_Request_codebook() | `SideAssetManager.fetchModel()` |
| `dash.all.debug.js` — send2DNNprocess() / Flask `/uploader` | `SRRuntimeStub.run()` (Phase 1) → `CodebookSREngine` (Phase 2) |
| `process.py` — decode+SR+encode multiprocess pipeline | Phase 1: MediaCodec(ExoPlayer) + `SRRuntimeStub` + `ImageView` |
| `dnn_appLocalServer.py` — Flask routes `/dnn`, `/imatrix` | `SideAssetManager` (HTTP fetch + local cache) |

## 4-Plane 구조

| Plane | 현재 (서버) | Phase 1 (Android) | Phase 2 (Android) |
|-------|------|------|------|
| Delivery/Playback | Browser + DASH.js + CDN | ExoPlayer DASH + CDN (동일) | + ABR (MPC HTTP) |
| Reconstruction | 서버 GPU process.py | `SRRuntimeStub` (bicubic 4x) | `CodebookSREngine` (PyTorch Mobile) |
| Runtime Integration | 서버 내부 파이프라인 | TextureView bitmap capture | JNI + PyTorch Mobile Lite |
| Measurement | PSNR, timing (서버) | SR latency ms/frame, frame save | + energy, memory profiling |

---

## Phase 1 — 구현 완료

```
android/ShortsCodebookPlayer/
├── Config.kt             — CDN URL, SEQUENCE 목록, SR 파라미터 상수
├── MainActivity.kt       — ExoPlayer DASH 재생 + 주기적 프레임 캡처 + SR stub 연결
├── SideAssetManager.kt   — imatrix.pt / decoder.pt / coarse_sr.pt CDN 다운 + 캐시
├── SRRuntimeStub.kt      — Bitmap.createScaledBitmap 4x (bicubic/nearest) + 레이턴시 측정
└── FrameSaver.kt         — LR + SR 프레임 JPEG 저장 (Pictures/ShortsCodebookSR/)
```

**동작 흐름**:
1. `Config.SEQUENCE[0]` MPD 로드 → ExoPlayer DASH 재생 (270×480 LR)
2. 병렬: `SideAssetManager` → imatrix.pt + decoder.pt + coarse_sr.pt 다운로드 → 로컬 캐시
3. 매 500ms: `TextureView.getBitmap()` → `SRRuntimeStub.run()` → `ImageView` 출력 (우측 SR, 좌측 LR)
4. 롱클릭: 현재 LR/SR 프레임 저장 → `Pictures/ShortsCodebookSR/`
5. 탭: 다음 영상으로 (watch_time 기반 자동 전환 예정)

**디버그 바**:
- tvStatus: 재생 상태 / 저장 메시지
- tvAssets: imatrix/decoder/coarse_sr 다운로드 완료 여부 (KB 표시)
- tvMetrics: SR 레이턴시 ms + 누적 평균 + 캡처 프레임 수

---

## Phase 2 — 예정

```
CodebookSREngine.kt
  — PyTorch Mobile Lite (.ptl) 로드 (decoder + coarse_sr)
  — ImatrixManager: imatrix.pt → short[] 파싱, frame_idx → indices
  — runSR(prevBitmap, currBitmap, frameIdx): Bitmap → Tensor → forward → Bitmap
  — latency 측정 + MetricsCollector 연동

MetricsCollector.kt
  — SystemClock.elapsedRealtimeNanos() → per-frame latency
  — ActivityManager.MemoryInfo → peak RSS
  — (실제 기기) BatteryManager → energy mJ/frame
  — (실제 기기) ThermalManager → throttling onset

model export (서버):
  python mobile/export_mobile_models.py
  → decoder_{class}.ptl + coarse_sr_{class}.ptl
```

`SRRuntimeStub` → `CodebookSREngine` 교체만으로 Phase 2 전환 가능하도록 인터페이스 통일 예정.

---

## Device Frame (실측 목표)

| Frame | Runtime | 기기 | 측정 지표 |
|-------|---------|------|---------|
| F1 | ExecuTorch + QNN HTP | Galaxy S24 (Snapdragon 8 Gen 3) | latency, PSNR, energy |
| F2 | PyTorch Mobile Lite (XNNPACK) | 모든 Android API 26+ | latency, PSNR |
| F3 | SNPE GPU float16 | Snapdragon 845 | NEMO 비교 기준 |
| F4 | LiteRT + NNAPI | T1/T2 | Google 벤치마크 호환 |

Phase 1 bicubic stub → F2 런타임으로 직접 교체.

---

## 빌드 방법

```
# Android Studio에서:
File → Open → mobile/android/ShortsCodebookPlayer
# 또는:
cd mobile/android/ShortsCodebookPlayer
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

CDN(163.152.162.202:8080)이 접근 가능한 네트워크에서 실행.
