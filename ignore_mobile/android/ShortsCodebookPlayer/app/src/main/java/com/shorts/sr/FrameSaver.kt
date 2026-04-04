package com.shorts.sr

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * SR 결과 프레임을 저장한다.
 * - API 29+: MediaStore (Pictures/ShortsCodebookSR/)
 * - API 26–28: /sdcard/Pictures/ShortsCodebookSR/ 직접 저장
 */
object FrameSaver {

    private const val TAG = "FrameSaver"
    private const val DIR = "ShortsCodebookSR"

    suspend fun save(
        context: Context,
        lrBitmap: Bitmap,
        hrBitmap: Bitmap,
        vid: String,
        frameIdx: Int,
        latencyMs: Long,
    ) = withContext(Dispatchers.IO) {
        val label = vid.replace("/", "_")
        saveBitmap(context, lrBitmap, "${label}_fr${frameIdx}_LR.jpg")
        saveBitmap(context, hrBitmap, "${label}_fr${frameIdx}_SR${latencyMs}ms.jpg")
        Log.i(TAG, "saved frame $frameIdx for $vid (SR latency=${latencyMs}ms)")
    }

    private fun saveBitmap(context: Context, bitmap: Bitmap, filename: String) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$DIR")
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: return
            context.contentResolver.openOutputStream(uri)?.use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
            }
        } else {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), DIR)
            dir.mkdirs()
            FileOutputStream(File(dir, filename)).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
            }
        }
    }
}
