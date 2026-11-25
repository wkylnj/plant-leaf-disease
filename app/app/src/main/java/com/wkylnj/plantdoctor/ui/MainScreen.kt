package com.wkylnj.plantdoctor.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.wkylnj.plantdoctor.data.LabelProb
import com.wkylnj.plantdoctor.data.UiState

private val Leaf = Color(0xFF246B4A)
private val LeafDark = Color(0xFF174A35)
private val Paper = Color(0xFFF5F7F2)
private val Ink = Color(0xFF19231D)
private val Muted = Color(0xFF59665E)
private val Line = Color(0xFFDCE4DB)
private val Warning = Color(0xFFA65312)
private val Danger = Color(0xFF9F3030)

@Composable
fun MainScreen(
    vm: MainViewModel,
    onPickCamera: () -> Unit,
    onPickGallery: () -> Unit,
    onOpenHistory: () -> Unit,
    onOpenAbout: () -> Unit,
    onOpenDetail: (String) -> Unit,
    onCorrect: (String) -> Unit,
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val health by vm.health.collectAsStateWithLifecycle()
    val feedbackDoneIds by vm.feedbackDoneIds.collectAsStateWithLifecycle()
    val feedbackError by vm.feedbackError.collectAsStateWithLifecycle()

    Surface(color = Paper, modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().height(60.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.End,
            ) {
                health?.let {
                    Text("${it.model} · ${it.classes} 类", color = Muted, fontSize = 12.sp)
                }
                IconButton(onClick = onOpenHistory) {
                    Icon(Icons.Filled.History, contentDescription = "识别记录", tint = LeafDark)
                }
                IconButton(onClick = onOpenAbout) {
                    Icon(Icons.Filled.Info, contentDescription = "关于叶片医生", tint = LeafDark)
                }
            }

            ImagePreview(state)

            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text("叶片医生", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = LeafDark)
                Text("拍一张叶片照片，查看可能的病害", fontSize = 15.sp, color = Muted)
            }

            when (state) {
                UiState.Idle -> PickActions(onPickCamera, onPickGallery)
                is UiState.Picked -> PickActions(onPickCamera, onPickGallery)
                is UiState.Uploading -> LoadingState()
                is UiState.Done -> {
                    val done = state as UiState.Done
                    PredictionResult(
                        top3 = done.prediction.top3,
                        elapsedMs = done.prediction.elapsedMs,
                        feedbackDone = done.prediction.requestId in feedbackDoneIds,
                        feedbackError = feedbackError,
                        onDetails = { onOpenDetail(done.prediction.requestId) },
                        onCorrect = { onCorrect(done.prediction.requestId) },
                        onConfirm = { vm.sendFeedback(done.prediction.requestId, true) },
                    )
                    OutlinedButton(
                        onClick = vm::reset,
                        modifier = Modifier.fillMaxWidth().height(50.dp),
                        shape = RoundedCornerShape(8.dp),
                    ) { Text("重新识别", fontSize = 16.sp) }
                }
                is UiState.Failed -> {
                    val failed = state as UiState.Failed
                    InfoBanner(failed.message, danger = true)
                    OutlinedButton(
                        onClick = { failed.path?.let { vm.recognize(java.io.File(it)) } },
                        modifier = Modifier.fillMaxWidth().height(50.dp),
                        shape = RoundedCornerShape(8.dp),
                    ) { Text("重试", fontSize = 16.sp) }
                    TextButton(onClick = vm::reset, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                        Text("换一张照片", color = Leaf)
                    }
                }
            }

            Spacer(Modifier.height(4.dp))
            TextButton(onClick = onOpenAbout, modifier = Modifier.fillMaxWidth()) {
                Text(
                    "结果仅供参考，不能替代专业诊断。用药请遵医嘱或咨询农技员。",
                    color = Muted,
                    fontSize = 13.sp,
                    lineHeight = 19.sp,
                )
            }
            Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable
private fun ImagePreview(state: UiState) {
    val path = when (state) {
        is UiState.Picked -> state.path
        is UiState.Uploading -> state.path
        is UiState.Done -> state.path
        is UiState.Failed -> state.path
        UiState.Idle -> null
    }
    Box(
        modifier = Modifier.fillMaxWidth().aspectRatio(1.45f).clip(RoundedCornerShape(8.dp))
            .background(Color(0xFFE5EBE4)),
        contentAlignment = Alignment.Center,
    ) {
        val bitmap = path?.let { remember(it) {
            runCatching {
                BitmapFactory.decodeFile(it, BitmapFactory.Options().apply { inSampleSize = 2 })
            }.getOrNull()
        } }
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "叶片照片",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Filled.PhotoLibrary, contentDescription = null, tint = Leaf, modifier = Modifier.size(40.dp))
                Text("选择一张清晰的叶片照片", color = Muted, fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun PickActions(onCamera: () -> Unit, onGallery: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Button(
            onClick = onCamera,
            modifier = Modifier.fillMaxWidth().height(54.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Leaf),
            shape = RoundedCornerShape(8.dp),
        ) {
            Icon(Icons.Filled.CameraAlt, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("拍照识别", fontSize = 16.sp)
        }
        OutlinedButton(
            onClick = onGallery,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(8.dp),
        ) {
            Icon(Icons.Filled.PhotoLibrary, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("从相册选择", fontSize = 16.sp)
        }
    }
}

@Composable
private fun LoadingState() {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CircularProgressIndicator(modifier = Modifier.size(22.dp), color = Leaf, strokeWidth = 2.dp)
        Spacer(Modifier.width(12.dp))
        Column {
            Text("正在识别…", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ink)
            Text("通常需要 1–3 秒", fontSize = 14.sp, color = Muted)
        }
    }
}

@Composable
private fun PredictionResult(
    top3: List<LabelProb>,
    elapsedMs: Double,
    feedbackDone: Boolean,
    feedbackError: String?,
    onDetails: () -> Unit,
    onCorrect: () -> Unit,
    onConfirm: () -> Unit,
) {
    val top = top3.firstOrNull() ?: return
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        if (top.prob < 0.6) {
            InfoBanner("模型不太确定，建议再拍一张或咨询农技员。", danger = false)
        }
        Surface(shape = RoundedCornerShape(8.dp), color = Color.White, tonalElevation = 1.dp) {
            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("最可能的结果", fontSize = 14.sp, color = Muted)
                Text(top.labelCn, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = LeafDark, lineHeight = 34.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    LinearProgressIndicator(
                        progress = { top.prob.toFloat().coerceIn(0f, 1f) },
                        modifier = Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(4.dp)),
                        color = if (top.prob >= 0.6) Leaf else Warning,
                        trackColor = Line,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text("${(top.prob * 100).toInt()}%", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ink)
                }
                Text("仅为模型置信度，不代表诊断准确率", fontSize = 13.sp, color = Muted)
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("其他可能", fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = Ink)
            top3.drop(1).forEach { candidate ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(candidate.labelCn, fontSize = 15.sp, color = Ink, modifier = Modifier.weight(1f))
                    Text("${(candidate.prob * 100).toInt()}%", fontSize = 15.sp, color = Muted)
                }
            }
            Text("推理耗时 ${elapsedMs.toInt()} ms", fontSize = 13.sp, color = Muted)
        }

        OutlinedButton(
            onClick = onDetails,
            modifier = Modifier.fillMaxWidth().height(48.dp),
            shape = RoundedCornerShape(8.dp),
        ) { Text("查看识别详情", fontSize = 15.sp, color = Leaf) }

        if (feedbackDone) {
            Text("感谢你的反馈", color = Leaf, fontSize = 14.sp)
        } else {
            Text("这个结果对吗？", color = Muted, fontSize = 14.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(onClick = onConfirm, modifier = Modifier.weight(1f).height(48.dp), shape = RoundedCornerShape(8.dp)) {
                    Text("是，结果正确", fontSize = 14.sp)
                }
                Button(
                    onClick = onCorrect,
                    modifier = Modifier.weight(1f).height(48.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Leaf),
                    shape = RoundedCornerShape(8.dp),
                ) { Text("结果不对", fontSize = 14.sp) }
            }
            feedbackError?.let { Text(it, color = Danger, fontSize = 13.sp) }
        }
    }
}

@Composable
internal fun InfoBanner(message: String, danger: Boolean) {
    val color = if (danger) Danger else Warning
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = color.copy(alpha = 0.1f),
        shape = RoundedCornerShape(8.dp),
    ) {
        Text(message, modifier = Modifier.padding(12.dp), color = color, fontSize = 14.sp, lineHeight = 20.sp)
    }
}
