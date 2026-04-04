package com.shorts.sr

import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.TextureView
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import com.shorts.sr.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

/**
 * Phase 1 — Main Activity
 *
 * player.js 대응:
 *   runPlayback(): sequence 순회, playPtr, QUEUE_LEN=5 prefetch
 *   → 여기서는 단순화: SEQUENCE 리스트 순회, prefetch 없음 (Phase 2에서 추가)
 *
 * dash.all.debug.js 대응:
 *   SuperPlayer.playNext() / attachSource() → ExoPlayer.setMediaItem() / prepare()
 *   send_DNN_Request_codebook() + triggerImatrixDownload() → SideAssetManager.fetchModel() + fetchImatrix()
 *   send2DNNprocess() / /uploader → SRRuntimeStub.runAndRecord() (Phase 1 stub)
 *
 * process.py 대응:
 *   decode_process + process_video_chunk + encode_process
 *   → Phase 1: MediaCodec decode (ExoPlayer) + SRRuntimeStub (bicubic) + ImageView 출력
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var player: ExoPlayer
    private lateinit var srStub: SRRuntimeStub
    private lateinit var assetManager: SideAssetManager

    private val captureHandler = Handler(Looper.getMainLooper())
    private var captureRunnable: Runnable? = null

    private var currentIdx = 0
    private var currentItem: VideoItem? = null
    private var assetState: SideAssetManager.AssetState? = null

    private var totalFramesCaptured = 0
    private var saveNextFrame = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        srStub = SRRuntimeStub()
        assetManager = SideAssetManager(this)

        setupPlayer()
        loadVideo(0)

        // 화면 탭 → 다음 영상
        binding.playerViewLR.setOnClickListener { advanceToNext() }

        // 볼륨 버튼(또는 버튼) 대신 간단히: 롱클릭 → 현재 프레임 저장
        binding.playerViewLR.setOnLongClickListener {
            saveNextFrame = true
            binding.tvStatus.text = "Saving next frame..."
            true
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  ExoPlayer setup
    //  dash.all.debug.js: SuperPlayer.setup() → MediaPlayer.initialize() 대응
    // ────────────────────────────────────────────────────────────────
    private fun setupPlayer() {
        player = ExoPlayer.Builder(this).build()
        binding.playerViewLR.player = player

        player.addListener(object : Player.Listener {
            override fun onPlaybackStateChanged(state: Int) {
                when (state) {
                    Player.STATE_READY   -> onPlayerReady()
                    Player.STATE_ENDED   -> advanceToNext()
                    Player.STATE_BUFFERING -> updateStatus("Buffering…")
                    else -> {}
                }
            }
        })
    }

    // ────────────────────────────────────────────────────────────────
    //  Video load — player.js: attachSource(mpd(pid)) 대응
    // ────────────────────────────────────────────────────────────────
    private fun loadVideo(idx: Int) {
        if (idx >= Config.SEQUENCE.size) {
            updateStatus("All videos done.")
            return
        }
        currentIdx = idx
        val item = Config.SEQUENCE[idx]
        currentItem = item

        Log.i(TAG, "[loadVideo] idx=$idx vid=${item.vid}")
        updateStatus("Loading: ${item.vid}")
        binding.tvAssets.text = "Assets: fetching…"
        binding.tvMetrics.text = "SR: —"
        binding.ivSR.setImageBitmap(null)
        srStub.resetStats()

        // ExoPlayer: LR DASH 재생
        // dash.all.debug.js: ManifestLoader → players[slot].attachSource(url) 대응
        val mediaItem = MediaItem.fromUri(item.mpdUrl())
        player.setMediaItem(mediaItem)
        player.prepare()
        player.playWhenReady = true

        // side asset 비동기 fetch
        // dash.all.debug.js: triggerImatrixDownload() + send_DNN_Request_codebook() 대응
        val state = SideAssetManager.AssetState(vid = item.vid, className = item.className)
        assetState = state

        lifecycleScope.launch {
            assetManager.fetchImatrix(item, state)
            updateAssetsStatus(state)
        }
        lifecycleScope.launch {
            assetManager.fetchModel(item, state)
            updateAssetsStatus(state)
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Player ready → start periodic frame capture + SR
    //  dash.all.debug.js: onFragmentLoadingCompleted → send2DNNprocess() 대응
    //  process.py: decode_process → data_queue → process_video_chunk 대응
    // ────────────────────────────────────────────────────────────────
    private fun onPlayerReady() {
        updateStatus("Playing LR (${Config.LR_W}×${Config.LR_H}) → SR stub (${Config.HR_W}×${Config.HR_H})")
        startCaptureCycle()
    }

    private fun startCaptureCycle() {
        stopCaptureCycle()
        val r = object : Runnable {
            override fun run() {
                captureAndSR()
                captureHandler.postDelayed(this, Config.CAPTURE_INTERVAL_MS)
            }
        }
        captureRunnable = r
        captureHandler.postDelayed(r, Config.CAPTURE_INTERVAL_MS)
    }

    private fun stopCaptureCycle() {
        captureRunnable?.let { captureHandler.removeCallbacks(it) }
        captureRunnable = null
    }

    // ────────────────────────────────────────────────────────────────
    //  Frame capture + SR stub
    //  SRRuntimeStub.run() = process.py: infer_with_imatrix() 자리 (Phase 1 bicubic)
    // ────────────────────────────────────────────────────────────────
    private fun captureAndSR() {
        val textureView = binding.playerViewLR.videoSurfaceView as? TextureView ?: return
        val lrBitmap: Bitmap = textureView.getBitmap(Config.LR_W, Config.LR_H) ?: return

        val result = srStub.runAndRecord(lrBitmap)
        totalFramesCaptured++

        binding.ivSR.setImageBitmap(result.output)
        binding.tvMetrics.text = "SR: ${result.latencyMs}ms (avg ${srStub.avgLatencyMs().toInt()}ms) | frames=$totalFramesCaptured"

        if (saveNextFrame) {
            saveNextFrame = false
            val vid = currentItem?.vid ?: return
            val idx = totalFramesCaptured
            lifecycleScope.launch {
                FrameSaver.save(this@MainActivity, lrBitmap, result.output, vid, idx, result.latencyMs)
                binding.tvStatus.text = "Frame $idx saved to Pictures/ShortsCodebookSR/"
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Advance — player.js: finish() → playPtr++ → player.playNext() 대응
    // ────────────────────────────────────────────────────────────────
    private fun advanceToNext() {
        stopCaptureCycle()
        val watchTime = currentItem?.watchTime ?: 0f
        Log.i(TAG, "[advance] vid=${currentItem?.vid} watchTime=$watchTime played=${player.currentPosition / 1000f}s")
        loadVideo(currentIdx + 1)
    }

    // ────────────────────────────────────────────────────────────────
    //  UI helpers
    // ────────────────────────────────────────────────────────────────
    private fun updateStatus(msg: String) {
        runOnUiThread { binding.tvStatus.text = msg }
    }

    private fun updateAssetsStatus(state: SideAssetManager.AssetState) {
        runOnUiThread {
            binding.tvAssets.text = buildString {
                append("imatrix: ${if (state.imatrixReady) "✓ ${state.imatrixPath?.length()?.div(1024)}KB" else "…"}")
                append("  decoder: ${if (state.decoderReady) "✓" else "…"}")
                append("  coarse_sr: ${if (state.coarseSrReady) "✓" else "…"}")
                if (state.allReady) append("  [ALL READY]")
            }
        }
    }

    override fun onPause() {
        super.onPause()
        stopCaptureCycle()
        player.pause()
    }

    override fun onResume() {
        super.onResume()
        if (player.playbackState == Player.STATE_READY) {
            player.play()
            startCaptureCycle()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopCaptureCycle()
        player.release()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
