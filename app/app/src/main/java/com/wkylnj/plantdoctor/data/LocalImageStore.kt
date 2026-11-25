package com.wkylnj.plantdoctor.data

import android.content.Context
import java.io.File

class LocalImageStore(context: Context) {
    private val directory = File(context.filesDir, "recognition-images").apply { mkdirs() }

    fun save(recordId: String, source: File): File? = runCatching {
        require(recordId.matches(Regex("[A-Za-z0-9_-]{1,64}")))
        val destination = File(directory, "$recordId.jpg")
        source.copyTo(destination, overwrite = true)
        destination
    }.getOrNull()

    fun find(recordId: String): File? = runCatching {
        require(recordId.matches(Regex("[A-Za-z0-9_-]{1,64}")))
        File(directory, "$recordId.jpg").takeIf { it.isFile }
    }.getOrNull()

    fun clear() {
        directory.listFiles()?.forEach { it.delete() }
    }
}
