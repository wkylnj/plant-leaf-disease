package com.wkylnj.plantdoctor.ui

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
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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

// ---------------------------------------------------------------- 设计基调
// 为什么用这套色：这是农业工具，不是社交产品。深绿主色给出"植物/生长"的
// 直觉，米白底降低户外强光下的眩光，橙色只留给"高置信度"的强调。
private val LeafGreen = Color(0xFF1F6F4A)
private val LeafGreenDark = Color(0xFF14472F)
private val Paper = Color(0xFFF7F5EF)
private val Ink = Color(0xFF1A1A1A)
private val Muted = Color(0xFF6B6B6B)
private val Warn = Color(0xFFB45309)
private val ErrRed = Color(0xFFB91C1C)

@Composable
fun MainScreen(vm: MainViewModel, onPickCamera: () -> Unit, onPickGallery: () -> Unit) {
    val state by vm.state.collectAsStateWithLifecycle()
    val health by vm.health.collectAsStateWithLifecycle()
    val fbSent by vm.feedbackSent.collectAsStateWithLifecycle()

    Surface(color = Paper, modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Header(health?.model)

            Preview(state)

            when (state) {
                is UiState.Idle -> PickButtons(onPickCamera, onPickGallery)
                is UiState.Uploading -> Busy("正在识别…")
                is UiState.Done -> {
                    val s = state as UiState.Done
                    ResultCard(s.prediction.top3, s.prediction.elapsedMs)
                    FeedbackRow(fbSent) { vm.sendFeedback(it) }
                    OutlinedButton(onClick = { vm.reset() }, modifier = Modifier.fillMaxWidth()) {
                        Text("重新拍一张")
                    }
                }
                is UiState.Failed -> {
                    ErrorCard((state as UiState.Failed).message)
                    OutlinedButton(onClick = { vm.reset() }, modifier = Modifier.fillMaxWidth()) {
                        Text("重试")
                    }
                }
                else -> Unit
            }

            Spacer(Modifier.height(4.dp))
            Text(
                "本应用仅用于课程演示，结果仅供参考，不能替代专业植保诊断。",
                fontSize = 11.sp,
                color = Muted,
                lineHeight = 16.sp,
            )
        }
    }
}

@Composable
private fun Header(modelName: String?) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("叶片医生", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = LeafGreenDark)
            Text("拍一张叶片照片，看看它得了什么病", fontSize = 13.sp, color = Muted)
        }
        if (modelName != null) {
            Box(
                Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .background(LeafGreen.copy(alpha = 0.12f))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(modelName, fontSize = 10.sp, color = LeafGreen)
            }
        }
    }
}

@Composable
private fun Preview(state: UiState) {
    val path = when (state) {
        is UiState.Picked -> state.path
        is UiState.Uploading -> state.path
        is UiState.Done -> state.path
        is UiState.Failed -> state.path
        else -> null
    }
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFFE8E4DA)),
        contentAlignment = Alignment.Center,
    ) {
        if (path != null) {
            val bmp = rememberBitmap(path)
            if (bmp != null) {
                androidx.compose.foundation.Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "待识别的叶片",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                )
            } else {
                Text("图片读取失败", color = Muted, fontSize = 13.sp)
            }
        } else {
            Text("还没有图片", color = Muted, fontSize = 13.sp)
        }
    }
}

@Composable
private fun rememberBitmap(path: String): android.graphics.Bitmap? {
    return androidx.compose.runtime.remember(path) {
        runCatching {
            val f = java.io.File(path)
            val opts = android.graphics.BitmapFactory.Options().apply { inSampleSize = 2 }
            android.graphics.BitmapFactory.decodeFile(f.absolutePath, opts)
        }.getOrNull()
    }
}

@Composable
private fun PickButtons(onCamera: () -> Unit, onGallery: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Button(
            onClick = onCamera,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            colors = ButtonDefaults.buttonColors(containerColor = LeafGreen),
            shape = RoundedCornerShape(12.dp),
        ) { Text("拍照识别", fontSize = 16.sp) }
        OutlinedButton(
            onClick = onGallery,
            modifier = Modifier.fillMaxWidth().height(48.dp),
            shape = RoundedCornerShape(12.dp),
        ) { Text("从相册选择", fontSize = 15.sp) }
    }
}

@Composable
private fun Busy(text: String) {
    Column(
        Modifier.fillMaxWidth().padding(vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(color = LeafGreen, modifier = Modifier.size(36.dp))
        Spacer(Modifier.height(12.dp))
        Text(text, fontSize = 14.sp, color = Muted)
    }
}

@Composable
private fun ResultCard(top3: List<LabelProb>, elapsedMs: Double) {
    val top = top3.firstOrNull() ?: return
    Card(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("识别结果", fontSize = 12.sp, color = Muted)
            Text(top.labelCn, fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Ink)
            Text(top.label, fontSize = 11.sp, color = Muted)

            val pct = (top.prob * 100).toInt()
            Row(verticalAlignment = Alignment.CenterVertically) {
                LinearProgressIndicator(
                    progress = { top.prob.toFloat() },
                    modifier = Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(4.dp)),
                    color = if (pct >= 70) LeafGreen else Warn,
                    trackColor = Color(0xFFEDEAE1),
                )
                Spacer(Modifier.width(10.dp))
                Text("$pct%", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Ink)
            }

            if (top3.size > 1) {
                Spacer(Modifier.height(4.dp))
                Text("其他可能", fontSize = 12.sp, color = Muted)
                top3.drop(1).forEach { other ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(other.labelCn, fontSize = 13.sp, color = Ink)
                        Text("${(other.prob * 100).toInt()}%", fontSize = 13.sp, color = Muted)
                    }
                }
            }
            Text("耗时 ${elapsedMs.toInt()} ms", fontSize = 10.sp, color = Muted)
        }
    }
}

@Composable
private fun FeedbackRow(sent: Boolean, onFeedback: (Boolean) -> Unit) {
    if (sent) {
        Text("感谢反馈", fontSize = 13.sp, color = LeafGreen)
        return
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("这个结果对吗？", fontSize = 13.sp, color = Muted, modifier = Modifier.weight(1f))
        TextButton(onClick = { onFeedback(true) }) { Text("对的", color = LeafGreen) }
        TextButton(onClick = { onFeedback(false) }) { Text("不对", color = ErrRed) }
    }
}

@Composable
private fun ErrorCard(message: String) {
    Card(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFDECEC)),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text("出错了", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = ErrRed)
            Spacer(Modifier.height(4.dp))
            Text(message, fontSize = 12.sp, color = ErrRed.copy(alpha = 0.85f), lineHeight = 17.sp)
        }
    }
}
