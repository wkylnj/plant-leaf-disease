package com.wkylnj.plantdoctor.data

/**
 * 一次识别的返回结果。
 *
 * 字段与后端 `POST /predict` 一一对应，**不要自己改名**——
 * 这是后端与安卓唯一的接口契约（docs/计划.md §2）。
 */
data class Prediction(
    val requestId: String,
    val top3: List<LabelProb>,
    val model: String,
    val elapsedMs: Double,
)

data class LabelProb(
    val label: String,
    val labelCn: String,
    val prob: Double,
)

/** 服务健康状态。 */
data class Health(
    val status: String,
    val model: String,
    val version: String,
    val classes: Int,
    val architecture: String = model,
    val imageSize: Int = 224,
)

data class HistoryRecord(
    val id: String,
    val createdAt: String,
    val top3: List<LabelProb>,
    val model: String,
    val elapsedMs: Double,
    val feedback: FeedbackRecord?,
    val imagePath: String? = null,
) {
    val top: LabelProb? get() = top3.firstOrNull()
}

data class HistoryPage(
    val items: List<HistoryRecord>,
    val total: Int,
)

data class FeedbackRecord(
    val isCorrect: Boolean,
    val trueLabel: String?,
    val createdAt: String?,
)

data class DiseaseClass(
    val label: String,
    val labelCn: String,
    val cropCn: String,
)

sealed interface HistoryState {
    data object Loading : HistoryState
    data class Loaded(val records: List<HistoryRecord>, val total: Int) : HistoryState
    data class Failed(val message: String) : HistoryState
}

sealed interface DetailState {
    data object Loading : DetailState
    data class Loaded(val record: HistoryRecord) : DetailState
    data object Missing : DetailState
    data class Failed(val message: String) : DetailState
}

sealed interface Screen {
    data object Home : Screen
    data object History : Screen
    data class Detail(val recordId: String, val from: Screen) : Screen
    data object About : Screen
}

/** 上传/识别过程中的界面状态。 */
sealed interface UiState {
    data object Idle : UiState

    /** 已选好图片但还没开始识别（当前流程会立刻进 Uploading，保留这个状态便于以后加预览确认）。 */
    data class Picked(val path: String) : UiState

    /** 带图片路径，否则识别中/识别完就没有图可显示了。 */
    data class Uploading(val path: String) : UiState

    data class Done(val path: String, val prediction: Prediction) : UiState

    data class Failed(val path: String?, val message: String) : UiState
}
