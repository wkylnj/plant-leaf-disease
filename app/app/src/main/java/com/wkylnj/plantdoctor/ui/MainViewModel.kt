package com.wkylnj.plantdoctor.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.wkylnj.plantdoctor.data.ApiClient
import com.wkylnj.plantdoctor.data.ApiException
import com.wkylnj.plantdoctor.data.Health
import com.wkylnj.plantdoctor.data.Prediction
import com.wkylnj.plantdoctor.data.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 主界面状态机。
 *
 * 职责很窄：持有当前图片、调用 ApiClient、把结果或错误写回 UiState。
 * 不碰 Compose，不碰 Context——这样能在没有 Android 环境的 JVM 单测里跑。
 */
class MainViewModel(private val api: ApiClient = ApiClient()) : ViewModel() {

    private val _state = MutableStateFlow<UiState>(UiState.Idle)
    val state: StateFlow<UiState> = _state.asStateFlow()

    private val _health = MutableStateFlow<Health?>(null)
    val health: StateFlow<Health?> = _health.asStateFlow()

    /** 记录当前展示的预测，反馈时要用它的 requestId。 */
    private var current: Prediction? = null

    private val _feedbackSent = MutableStateFlow(false)
    val feedbackSent: StateFlow<Boolean> = _feedbackSent.asStateFlow()

    fun checkHealth() {
        viewModelScope.launch {
            _health.value = withContext(Dispatchers.IO) {
                runCatching { api.health() }.getOrNull()
            }
        }
    }

    /** 用户选好图片后调用，立刻开始识别。 */
    fun recognize(file: File) {
        _feedbackSent.value = false
        val path = file.absolutePath
        _state.value = UiState.Uploading(path)
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { api.predict(file) }
            }
            result.fold(
                onSuccess = {
                    current = it
                    _state.value = UiState.Done(path, it)
                },
                onFailure = { e ->
                    val msg = when (e) {
                        is ApiException -> "服务返回 ${e.code}：${e.message}"
                        is java.net.ConnectException -> "连不上服务器，检查后端是否启动、地址是否对"
                        is java.net.SocketTimeoutException -> "请求超时，图片可能太大或网络太慢"
                        else -> e.message ?: e.javaClass.simpleName
                    }
                    _state.value = UiState.Failed(path, msg)
                },
            )
        }
    }

    fun sendFeedback(isCorrect: Boolean) {
        val p = current ?: return
        if (_feedbackSent.value) return
        _feedbackSent.value = true
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                runCatching { api.feedback(p.requestId, isCorrect) }
            }
        }
    }

    fun reset() {
        current = null
        _feedbackSent.value = false
        _state.value = UiState.Idle
    }
}
