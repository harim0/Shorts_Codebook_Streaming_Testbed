package com.shorts.sr

object Config {
    const val CDN_BASE = "http://163.152.162.202:8080"
    const val ABR_BASE = "http://163.152.162.202:8334"

    // sequence.json의 일부 — Phase 1 고정 목록
    val SEQUENCE = listOf(
        VideoItem("Animal/SDR_Animal_23rp",  "Animal", 5.0f, 3.75f),
        VideoItem("Animal/SDR_Animal_3k7l",  "Animal", 5.0f, 3.75f),
        VideoItem("Dance/SDR_Dance_abcd",    "Dance",  5.0f, 3.75f),
    )

    // SR: LR → HR 스케일 비율
    const val SR_SCALE = 4          // 270→1080, 480→1920
    const val LR_W = 270
    const val LR_H = 480
    const val HR_W = LR_W * SR_SCALE   // 1080
    const val HR_H = LR_H * SR_SCALE   // 1920

    // 프레임 캡처 주기 (ms) — Phase 1 stub
    const val CAPTURE_INTERVAL_MS = 500L

    // 로컬 캐시 디렉토리 이름
    const val CACHE_DIR_IMATRIX = "imatrix"
    const val CACHE_DIR_MODEL   = "model"
}

data class VideoItem(
    val vid: String,        // e.g. "Animal/SDR_Animal_23rp"
    val className: String,  // e.g. "Animal"
    val duration: Float,
    val watchTime: Float,
) {
    fun mpdUrl() = "${Config.CDN_BASE}/dash/data/$vid/multi_resolution.mpd"
    fun imatrixUrl() = "${Config.CDN_BASE}/dash/data/$vid/imatrix.pt"
    fun decoderUrl() = "${Config.CDN_BASE}/dash/model/$className/decoder.pt"
    fun coarseSrUrl() = "${Config.CDN_BASE}/dash/model/$className/coarse_sr.pt"
}
