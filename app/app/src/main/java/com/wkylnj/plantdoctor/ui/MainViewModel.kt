package com.wkylnj.plantdoctor.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.wkylnj.plantdoctor.data.ApiClient
import com.wkylnj.plantdoctor.data.ApiException
import com.wkylnj.plantdoctor.data.DetailState
import com.wkylnj.plantdoctor.data.DiseaseClass
import com.wkylnj.plantdoctor.data.Health
import com.wkylnj.plantdoctor.data.HistoryRecord
import com.wkylnj.plantdoctor.data.HistoryState
import com.wkylnj.plantdoctor.data.LocalImageStore
import com.wkylnj.plantdoctor.data.Prediction
import com.wkylnj.plantdoctor.data.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val api = ApiClient(application)
    private val imageStore = LocalImageStore(application)

    private val _state = MutableStateFlow<UiState>(UiState.Idle)
    val state: StateFlow<UiState> = _state.asStateFlow()

    private val _health = MutableStateFlow<Health?>(null)
    val health: StateFlow<Health?> = _health.asStateFlow()

    private val _history = MutableStateFlow<HistoryState>(HistoryState.Loaded(emptyList(), 0))
    val history: StateFlow<HistoryState> = _history.asStateFlow()

    private val _detail = MutableStateFlow<DetailState>(DetailState.Loading)
    val detail: StateFlow<DetailState> = _detail.asStateFlow()

    private val _classes = MutableStateFlow<List<DiseaseClass>>(emptyList())
    val classes: StateFlow<List<DiseaseClass>> = _classes.asStateFlow()

    private val _feedbackDoneIds = MutableStateFlow<Set<String>>(emptySet())
    val feedbackDoneIds: StateFlow<Set<String>> = _feedbackDoneIds.asStateFlow()

    private val _feedbackBusy = MutableStateFlow(false)
    val feedbackBusy: StateFlow<Boolean> = _feedbackBusy.asStateFlow()

    private val _feedbackError = MutableStateFlow<String?>(null)
    val feedbackError: StateFlow<String?> = _feedbackError.asStateFlow()

    private val _clearingHistory = MutableStateFlow(false)
    val clearingHistory: StateFlow<Boolean> = _clearingHistory.asStateFlow()

    private val _historyError = MutableStateFlow<String?>(null)
    val historyError: StateFlow<String?> = _historyError.asStateFlow()

    private val _historyLoadingMore = MutableStateFlow(false)
    val historyLoadingMore: StateFlow<Boolean> = _historyLoadingMore.asStateFlow()

    fun checkHealth() {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { api.health() } }
            _health.value = result.getOrNull()
            if (result.isSuccess) {
                _classes.value = withContext(Dispatchers.IO) {
                    runCatching { api.classes() }.getOrDefault(emptyList())
                }
            }
        }
    }

    fun recognize(file: File) {
        _feedbackError.value = null
        _state.value = UiState.Uploading(file.absolutePath)
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { api.predict(file) } }
            result.fold(
                onSuccess = { prediction ->
                    val saved = withContext(Dispatchers.IO) { imageStore.save(prediction.requestId, file) }
                    val displayFile = saved ?: file
                    if (saved != null && saved != file) file.delete()
                    _state.value = UiState.Done(displayFile.absolutePath, prediction)
                    refreshHistory()
                },
                onFailure = { error ->
                    _state.value = UiState.Failed(file.absolutePath, messageFor(error))
                },
            )
        }
    }

    fun refreshHistory() {
        _history.value = HistoryState.Loading
        _historyError.value = null
        _historyLoadingMore.value = false
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val page = api.history()
                    page.copy(items = page.items.map { record ->
                        record.copy(imagePath = imageStore.find(record.id)?.absolutePath)
                    })
                }
            }
            _history.value = result.fold(
                onSuccess = { HistoryState.Loaded(it.items, it.total) },
                onFailure = { HistoryState.Failed(messageFor(it)) },
            )
        }
    }

    fun loadMoreHistory() {
        val loaded = _history.value as? HistoryState.Loaded ?: return
        if (_historyLoadingMore.value || loaded.records.size >= loaded.total) return
        _historyLoadingMore.value = true
        _historyError.value = null
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val page = api.history(offset = loaded.records.size)
                    page.copy(items = page.items.map { record ->
                        record.copy(imagePath = imageStore.find(record.id)?.absolutePath)
                    })
                }
            }
            _historyLoadingMore.value = false
            result.fold(
                onSuccess = { page ->
                    val latest = _history.value as? HistoryState.Loaded ?: loaded
                    _history.value = HistoryState.Loaded(latest.records + page.items, page.total)
                },
                onFailure = { _historyError.value = messageFor(it) },
            )
        }
    }

    fun loadRecord(recordId: String) {
        _detail.value = DetailState.Loading
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    api.historyRecord(recordId)?.let { record ->
                        record.copy(imagePath = imageStore.find(record.id)?.absolutePath)
                    }
                }
            }
            _detail.value = result.fold(
                onSuccess = { record -> if (record == null) DetailState.Missing else DetailState.Loaded(record) },
                onFailure = { DetailState.Failed(messageFor(it)) },
            )
        }
    }

    fun sendFeedback(requestId: String, isCorrect: Boolean, trueLabel: String? = null) {
        if (_feedbackBusy.value || requestId in _feedbackDoneIds.value) return
        _feedbackBusy.value = true
        _feedbackError.value = null
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { api.feedback(requestId, isCorrect, trueLabel) }
            }
            _feedbackBusy.value = false
            result.fold(
                onSuccess = {
                    _feedbackDoneIds.value = _feedbackDoneIds.value + requestId
                    refreshHistory()
                    loadRecord(requestId)
                },
                onFailure = { _feedbackError.value = messageFor(it) },
            )
        }
    }

    fun beginFeedback() {
        _feedbackError.value = null
    }

    fun clearHistory() {
        if (_clearingHistory.value) return
        _clearingHistory.value = true
        _historyError.value = null
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val count = api.clearHistory()
                    imageStore.clear()
                    count
                }
            }
            _clearingHistory.value = false
            result.fold(
                onSuccess = {
                    _history.value = HistoryState.Loaded(emptyList(), 0)
                    _feedbackDoneIds.value = emptySet()
                    _state.value = UiState.Idle
                },
                onFailure = { _historyError.value = messageFor(it) },
            )
        }
    }

    fun reset() {
        _feedbackError.value = null
        _state.value = UiState.Idle
    }

    private fun messageFor(error: Throwable): String = when (error) {
        is ApiException -> when (error.code) {
            404 -> "这条记录已不存在，请刷新后再试"
            413 -> "图片太大，请换一张较小的照片"
            else -> "服务暂时不可用，请稍后重试"
        }
        is IOException, is java.net.SocketTimeoutException -> "网络好像不通，检查一下再试"
        else -> "操作没有完成，请稍后重试"
    }
}
