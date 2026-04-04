package com.shorts.sr

import android.graphics.Bitmap
import android.util.Log

/**
 * Phase 1 SR Runtime Stub
 *
 * process.py 대응:
 *   process_video_chunk(): LR frame → SR → HR frame
 *   여기서는 bicubic(Android Bitmap.createScaledBitmap(filter=true)) 또는
 *   nearest-neighbor(filter=false)로 4x 업스케일 수행.
 *
 * Phase 2에서 이 클래스를 CodebookSREngine(PyTorch Mobile Lite)으로 교체.
 */
class SRRuntimeStub {

    enum class Mode {
        BICUBIC,       // Bitmap.createScaledBitmap(filter=true) — bilinear/bicubic-ish
        NEAREST,       // Bitmap.createScaledBitmap(filter=false) — 최저 레이턴시 기준선
    }

    var mode = Mode.BICUBIC

    data class SRResult(
        val output: Bitmap,     // HR 결과 (1080×1920 or targetW×targetH)
        val latencyMs: Long,    // SR 소요 시간
        val inputW: Int,
        val inputH: Int,
        val outputW: Int,
        val outputH: Int,
    )

    /**
     * LR bitmap을 4x 업스케일해 반환한다.
     * @param lr   LR 프레임 (270×480 portrait)
     * @param scale 업스케일 비율 (기본 4x)
     */
    fun run(lr: Bitmap, scale: Int = Config.SR_SCALE): SRResult {
        val targetW = lr.width * scale
        val targetH = lr.height * scale

        val t0 = System.currentTimeMillis()
        val hr = Bitmap.createScaledBitmap(lr, targetW, targetH, mode == Mode.BICUBIC)
        val latency = System.currentTimeMillis() - t0

        Log.d(TAG, "[SRStub] ${lr.width}×${lr.height} → ${targetW}×${targetH} | ${latency}ms | mode=$mode")

        return SRResult(
            output    = hr,
            latencyMs = latency,
            inputW    = lr.width,
            inputH    = lr.height,
            outputW   = targetW,
            outputH   = targetH,
        )
    }

    /**
     * 누적 측정: N프레임 평균 레이턴시
     */
    private val latencyHistory = ArrayDeque<Long>(100)

    fun runAndRecord(lr: Bitmap): SRResult {
        val result = run(lr)
        latencyHistory.addLast(result.latencyMs)
        if (latencyHistory.size > 100) latencyHistory.removeFirst()
        return result
    }

    fun avgLatencyMs(): Float =
        if (latencyHistory.isEmpty()) 0f
        else latencyHistory.average().toFloat()

    fun resetStats() = latencyHistory.clear()

    companion object {
        private const val TAG = "SRRuntimeStub"
    }
}
