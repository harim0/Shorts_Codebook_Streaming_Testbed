package com.shorts.sr

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * CDN에서 side asset을 다운로드하고 로컬 캐시에 저장한다.
 *
 * side assets:
 *   - imatrix.pt   : /dash/data/{class}/{vid}/imatrix.pt   (~9.3MB per video)
 *   - decoder.pt   : /dash/model/{class}/decoder.pt        (~880KB per class)
 *   - coarse_sr.pt : /dash/model/{class}/coarse_sr.pt      (~929KB per class)
 *
 * player.js 대응:
 *   makeDnnCtx() + classDNNReady 추적 → 여기서는 AssetState + callback으로 대응
 * dash.all.debug.js 대응:
 *   triggerImatrixDownload() + send_DNN_Request_codebook() → fetchImatrix() + fetchModel()
 */
class SideAssetManager(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    data class AssetState(
        val vid: String,
        val className: String,
        var imatrixReady: Boolean = false,
        var decoderReady: Boolean = false,
        var coarseSrReady: Boolean = false,
        var imatrixPath: File? = null,
        var decoderPath: File? = null,
        var coarseSrPath: File? = null,
        // timing
        var imatrixMs: Long = 0L,
        var decoderMs: Long = 0L,
        var coarseSrMs: Long = 0L,
    ) {
        val modelReady get() = decoderReady && coarseSrReady
        val allReady get() = imatrixReady && modelReady
    }

    /** imatrix.pt 다운로드. 이미 캐시되어 있으면 즉시 반환. */
    suspend fun fetchImatrix(item: VideoItem, state: AssetState): File? = withContext(Dispatchers.IO) {
        val cacheDir = File(context.cacheDir, Config.CACHE_DIR_IMATRIX).also { it.mkdirs() }
        val localFile = File(cacheDir, item.vid.replace("/", "_") + "_imatrix.pt")

        if (localFile.exists() && localFile.length() > 0) {
            Log.d(TAG, "[imatrix] cache hit: ${localFile.name} (${localFile.length() / 1024}KB)")
            state.imatrixReady = true
            state.imatrixPath = localFile
            return@withContext localFile
        }

        val t0 = System.currentTimeMillis()
        Log.d(TAG, "[imatrix] fetching ${item.imatrixUrl()}")
        val result = download(item.imatrixUrl(), localFile)
        state.imatrixMs = System.currentTimeMillis() - t0

        if (result != null) {
            state.imatrixReady = true
            state.imatrixPath = result
            Log.i(TAG, "[imatrix] done: ${result.length() / 1024}KB in ${state.imatrixMs}ms")
        } else {
            Log.e(TAG, "[imatrix] FAILED for ${item.vid}")
        }
        result
    }

    /** decoder.pt + coarse_sr.pt 다운로드. 클래스당 1회. */
    suspend fun fetchModel(item: VideoItem, state: AssetState) = withContext(Dispatchers.IO) {
        val cacheDir = File(context.cacheDir, Config.CACHE_DIR_MODEL).also { it.mkdirs() }

        listOf(
            Triple("decoder.pt",   item.decoderUrl(),   "decoderReady"),
            Triple("coarse_sr.pt", item.coarseSrUrl(),  "coarseSrReady"),
        ).forEach { (name, url, stateKey) ->
            val localFile = File(cacheDir, "${item.className}_$name")
            if (localFile.exists() && localFile.length() > 0) {
                Log.d(TAG, "[model] cache hit: $name")
                applyModelState(state, stateKey, localFile)
                return@forEach
            }
            val t0 = System.currentTimeMillis()
            Log.d(TAG, "[model] fetching $url")
            val result = download(url, localFile)
            val ms = System.currentTimeMillis() - t0
            if (result != null) {
                applyModelState(state, stateKey, result)
                Log.i(TAG, "[model] $name done: ${result.length() / 1024}KB in ${ms}ms")
            } else {
                Log.e(TAG, "[model] FAILED: $name")
            }
        }
    }

    private fun applyModelState(state: AssetState, key: String, file: File) {
        when (key) {
            "decoderReady"  -> { state.decoderReady = true;  state.decoderPath = file }
            "coarseSrReady" -> { state.coarseSrReady = true; state.coarseSrPath = file }
        }
    }

    private fun download(url: String, dest: File): File? {
        return try {
            val req = Request.Builder().url(url).build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    Log.e(TAG, "HTTP ${resp.code} for $url")
                    return null
                }
                resp.body!!.byteStream().use { input ->
                    dest.outputStream().use { out -> input.copyTo(out) }
                }
                dest
            }
        } catch (e: Exception) {
            Log.e(TAG, "download error $url: ${e.message}")
            null
        }
    }

    companion object {
        private const val TAG = "SideAssetManager"
    }
}
