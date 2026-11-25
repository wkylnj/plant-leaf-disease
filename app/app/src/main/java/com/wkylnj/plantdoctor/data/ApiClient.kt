package com.wkylnj.plantdoctor.data

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * 后端客户端。
 *
 * 为什么不用 Retrofit + OkHttp：
 * 本机 Gradle 缓存里没有这两个库，Alpha 阶段网络又不稳定，
 * 用系统自带的 HttpURLConnection 手写 multipart 反而更快跑通。
 * 代价是代码多 60 行，好处是零第三方依赖、包体积不变。
 *
 * 注意：所有网络调用都必须在 IO 线程执行（调用方用 Dispatchers.IO 包）。
 */
class ApiClient(context: Context, private val baseUrl: String = BuildConfigApi.base) {
    private val deviceId: String = File(context.noBackupFilesDir, "device-id").let { file ->
        file.takeIf { it.isFile }?.readText()?.takeIf { it.isNotBlank() }
            ?: UUID.randomUUID().toString().replace("-", "").also { file.writeText(it) }
    }

    /** 检查服务是否在线。 */
    fun health(): Health {
        send("POST", "/users/$deviceId", null, null)
        val text = send("GET", "/health", null, null)
        val o = JSONObject(text)
        return Health(
            status = o.optString("status", "?"),
            model = o.optString("model", "?"),
            version = o.optString("version", "?"),
            classes = o.optInt("classes", 0),
            architecture = o.optString("architecture", o.optString("model", "?")),
            imageSize = o.optInt("img_size", 224),
        )
    }

    /** 上传图片并拿回 top-3。 */
    fun predict(image: File): Prediction {
        val boundary = "----plantdoctor" + UUID.randomUUID().toString().replace("-", "")
        val text = send("POST", "/predict", image.readBytes(), boundary)
        val o = JSONObject(text)
        val arr = o.getJSONArray("top3")
        val list = ArrayList<LabelProb>(arr.length())
        for (i in 0 until arr.length()) {
            val it = arr.getJSONObject(i)
            list.add(
                LabelProb(
                    label = it.getString("label"),
                    labelCn = it.optString("label_cn", it.getString("label")),
                    prob = it.getDouble("prob"),
                )
            )
        }
        return Prediction(
            requestId = o.getString("request_id"),
            top3 = list,
            model = o.optString("model", "?"),
            elapsedMs = o.optDouble("elapsed_ms", 0.0),
        )
    }

    /** 提交用户反馈。 */
    fun feedback(requestId: String, isCorrect: Boolean, trueLabel: String? = null) {
        val body = JSONObject().apply {
            put("request_id", requestId)
            put("is_correct", isCorrect)
            if (trueLabel != null) put("true_label", trueLabel)
        }.toString()
        send("POST", "/feedback", body.toByteArray(Charsets.UTF_8), null, isJson = true)
    }

    fun history(limit: Int = 100, offset: Int = 0): HistoryPage {
        val obj = JSONObject(send("GET", "/history?limit=$limit&offset=$offset", null, null))
        val arr = obj.optJSONArray("items") ?: org.json.JSONArray()
        val items = (0 until arr.length()).mapNotNull { index ->
            arr.optJSONObject(index)?.let(::parseHistoryRecord)
        }
        return HistoryPage(items, obj.optInt("total", items.size))
    }

    fun historyRecord(requestId: String): HistoryRecord? {
        val text = runCatching { send("GET", "/history/$requestId", null, null) }
            .getOrElse { error -> if (error is ApiException && error.code == 404) return null else throw error }
        return parseHistoryRecord(JSONObject(text))
    }

    fun clearHistory(): Int {
        val obj = JSONObject(send("DELETE", "/history", null, null))
        return obj.optInt("deleted", 0)
    }

    fun classes(): List<DiseaseClass> {
        val arr = org.json.JSONArray(send("GET", "/classes", null, null))
        return (0 until arr.length()).mapNotNull { index ->
            arr.optJSONObject(index)?.let {
                DiseaseClass(
                    label = it.optString("label"),
                    labelCn = it.optString("label_cn"),
                    cropCn = it.optString("crop_cn", "其他"),
                )
            }
        }
    }

    private fun parseHistoryRecord(obj: JSONObject): HistoryRecord {
        val labels = obj.optJSONArray("top3")
        val top3 = if (labels == null) emptyList() else (0 until labels.length()).mapNotNull { index ->
            labels.optJSONObject(index)?.let {
                LabelProb(
                    label = it.optString("label"),
                    labelCn = it.optString("label_cn", it.optString("label")),
                    prob = it.optDouble("prob", 0.0),
                )
            }
        }
        val fb = obj.optJSONObject("feedback")?.let {
            FeedbackRecord(
                isCorrect = it.optBoolean("is_correct"),
                trueLabel = it.optString("true_label").takeIf { value -> value.isNotEmpty() },
                createdAt = it.optString("created_at").takeIf { value -> value.isNotEmpty() },
            )
        }
        return HistoryRecord(
            id = obj.optString("id"),
            createdAt = obj.optString("created_at"),
            top3 = top3,
            model = obj.optString("model_name", "?"),
            elapsedMs = obj.optDouble("elapsed_ms", 0.0),
            feedback = fb,
        )
    }

    // ------------------------------------------------------------ 底层
    private fun send(
        method: String,
        path: String,
        body: ByteArray?,
        boundary: String?,
        isJson: Boolean = false,
    ): String {
        val conn = (URL(baseUrl.trimEnd('/') + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 8000
            readTimeout = 30000
            doInput = true
            if (body != null) doOutput = true
        }
        try {
            conn.setRequestProperty("X-Device-ID", deviceId)
            if (body != null) {
                if (boundary != null) {
                    conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
                } else if (isJson) {
                    conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                }
                DataOutputStream(conn.outputStream).use { out ->
                    if (boundary != null) {
                        // 单文件字段，字段名固定为 file（后端 UploadFile 参数名）
                        out.writeBytes("--$boundary\r\n")
                        out.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"leaf.jpg\"\r\n")
                        out.writeBytes("Content-Type: image/jpeg\r\n\r\n")
                        out.write(body)
                        out.writeBytes("\r\n--$boundary--\r\n")
                    } else {
                        out.write(body)
                    }
                    out.flush()
                }
            }

            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader()?.use(BufferedReader::readText).orEmpty()
            if (code !in 200..299) {
                val detail = runCatching { JSONObject(text).optString("detail", text) }.getOrDefault(text)
                throw ApiException(code, detail.ifBlank { "HTTP $code" })
            }
            return text
        } finally {
            conn.disconnect()
        }
    }
}

class ApiException(val code: Int, message: String) : Exception(message)

/** 把 BuildConfig 的常量单独包一层，方便单测时替换。 */
object BuildConfigApi {
    val base: String get() = com.wkylnj.plantdoctor.BuildConfig.API_BASE
}
