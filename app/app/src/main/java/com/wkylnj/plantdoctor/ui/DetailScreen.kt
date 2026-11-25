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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
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
import com.wkylnj.plantdoctor.data.DetailState
import com.wkylnj.plantdoctor.data.HistoryRecord

private val DetailGreen = Color(0xFF246B4A)
private val DetailInk = Color(0xFF19231D)
private val DetailMuted = Color(0xFF59665E)
private val DetailPaper = Color(0xFFF5F7F2)

@Composable
fun DetailScreen(
    vm: MainViewModel,
    onBack: () -> Unit,
    onCorrect: (String) -> Unit,
) {
    val state by vm.detail.collectAsStateWithLifecycle()
    val feedbackDoneIds by vm.feedbackDoneIds.collectAsStateWithLifecycle()
    val feedbackError by vm.feedbackError.collectAsStateWithLifecycle()

    Surface(color = DetailPaper, modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier.fillMaxWidth().height(60.dp).padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = DetailInk) }
                Text("识别详情", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = DetailInk)
            }
            when (state) {
                DetailState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = DetailGreen)
                }
                DetailState.Missing -> DetailMessage(
                    title = "记录不存在",
                    body = "这条记录可能已经被清空。",
                    onAction = onBack,
                )
                is DetailState.Failed -> DetailMessage(
                    title = "记录读取失败",
                    body = (state as DetailState.Failed).message,
                    onAction = onBack,
                )
                is DetailState.Loaded -> DetailRecord(
                    record = (state as DetailState.Loaded).record,
                    feedbackDone = (state as DetailState.Loaded).record.id in feedbackDoneIds ||
                        (state as DetailState.Loaded).record.feedback != null,
                    feedbackError = feedbackError,
                    onCorrect = { onCorrect((state as DetailState.Loaded).record.id) },
                    onConfirm = { vm.sendFeedback((state as DetailState.Loaded).record.id, true) },
                )
            }
        }
    }
}

@Composable
private fun DetailRecord(
    record: HistoryRecord,
    feedbackDone: Boolean,
    feedbackError: String?,
    onCorrect: () -> Unit,
    onConfirm: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        val bitmap = record.imagePath?.let { remember(it) { runCatching { BitmapFactory.decodeFile(it) }.getOrNull() } }
        Box(
            Modifier.fillMaxWidth().aspectRatio(4f / 3f).clip(RoundedCornerShape(8.dp)).background(Color(0xFFE3EAE2)),
            contentAlignment = Alignment.Center,
        ) {
            if (bitmap != null) {
                Image(bitmap.asImageBitmap(), contentDescription = "叶片原图", modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            } else {
                Text("本机照片不可用", color = DetailMuted, fontSize = 14.sp)
            }
        }

        val top = record.top
        if (top != null) {
            if (top.prob < 0.6) InfoBanner("模型不太确定，建议再拍一张或咨询农技员。", danger = false)
            Text(top.labelCn, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = DetailGreen, lineHeight = 34.sp)
            Text("${(top.prob * 100).toInt()}% 模型置信度", fontSize = 15.sp, color = DetailMuted)
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("其他可能", fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = DetailInk)
            record.top3.drop(1).forEach { candidate ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(candidate.labelCn, fontSize = 15.sp, color = DetailInk)
                    Text("${(candidate.prob * 100).toInt()}%", fontSize = 15.sp, color = DetailMuted)
                }
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("识别信息", fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = DetailInk)
            DetailValue("时间", formatRecordDate(record.createdAt))
            DetailValue("模型", record.model)
            DetailValue("耗时", "${record.elapsedMs.toInt()} ms")
            DetailValue("反馈", record.feedback?.let {
                if (it.isCorrect) "已确认结果正确" else it.trueLabel?.let { label -> "已纠错：$label" } ?: "已反馈识别错误"
            } ?: "未反馈")
        }

        if (!feedbackDone) {
            Button(onClick = onCorrect, modifier = Modifier.fillMaxWidth().height(50.dp), shape = RoundedCornerShape(8.dp)) {
                Text("识别不对，我要纠错", fontSize = 15.sp)
            }
            OutlinedButton(onClick = onConfirm, modifier = Modifier.fillMaxWidth().height(48.dp), shape = RoundedCornerShape(8.dp)) {
                Text("结果正确", fontSize = 15.sp)
            }
        } else {
            Text("感谢你的反馈", color = DetailGreen, fontSize = 14.sp)
        }
        feedbackError?.let { Text(it, color = Color(0xFF9F3030), fontSize = 13.sp) }
        Text("结果仅供参考，不能替代专业诊断。", color = DetailMuted, fontSize = 13.sp)
        Spacer(Modifier.height(12.dp))
    }
}

@Composable
private fun DetailValue(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = DetailMuted, fontSize = 14.sp)
        Spacer(Modifier.width(18.dp))
        Text(value, color = DetailInk, fontSize = 14.sp)
    }
}

@Composable
private fun DetailMessage(title: String, body: String, onAction: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(title, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = DetailInk)
        Spacer(Modifier.height(8.dp))
        Text(body, fontSize = 14.sp, color = DetailMuted)
        Spacer(Modifier.height(16.dp))
        OutlinedButton(onClick = onAction) { Text("返回") }
    }
}
