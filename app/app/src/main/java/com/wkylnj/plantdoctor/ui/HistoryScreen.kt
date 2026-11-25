package com.wkylnj.plantdoctor.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
import com.wkylnj.plantdoctor.data.HistoryRecord
import com.wkylnj.plantdoctor.data.HistoryState
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter

private val HistoryGreen = Color(0xFF246B4A)
private val HistoryInk = Color(0xFF19231D)
private val HistoryMuted = Color(0xFF59665E)
private val HistoryPaper = Color(0xFFF5F7F2)

@Composable
fun HistoryScreen(vm: MainViewModel, onBack: () -> Unit, onOpenDetail: (String) -> Unit, onGoHome: () -> Unit) {
    val state by vm.history.collectAsStateWithLifecycle()
    val clearing by vm.clearingHistory.collectAsStateWithLifecycle()
    val loadingMore by vm.historyLoadingMore.collectAsStateWithLifecycle()
    val error by vm.historyError.collectAsStateWithLifecycle()
    var confirmClear by remember { mutableStateOf(false) }
    val loaded = state as? HistoryState.Loaded
    val records = loaded?.records.orEmpty()
    val total = loaded?.total ?: 0

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { if (!clearing) confirmClear = false },
            title = { Text("清空识别记录？") },
            text = { Text("将删除这台设备上的 $total 条记录和本机保存的照片，此操作不可恢复。") },
            confirmButton = {
                TextButton(
                    enabled = !clearing,
                    onClick = { confirmClear = false; vm.clearHistory() },
                ) { Text("清空", color = Color(0xFF9F3030)) }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("取消") } },
        )
    }

    Surface(color = HistoryPaper, modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier.fillMaxWidth().height(64.dp).padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = HistoryInk) }
                Column(Modifier.weight(1f)) {
                    Text("识别记录", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = HistoryInk)
                    Text("共 $total 条", fontSize = 13.sp, color = HistoryMuted)
                }
                if (records.isNotEmpty()) {
                    IconButton(onClick = { confirmClear = true }, enabled = !clearing) {
                        Icon(Icons.Filled.DeleteOutline, contentDescription = "清空记录", tint = Color(0xFF9F3030))
                    }
                }
            }

            error?.let { InfoBanner(it, danger = true) }
            when (state) {
                HistoryState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = HistoryGreen)
                }
                is HistoryState.Failed -> Column(
                    Modifier.fillMaxSize().padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    Text("记录读取失败", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = HistoryInk)
                    Spacer(Modifier.height(8.dp))
                    Text((state as HistoryState.Failed).message, color = HistoryMuted, fontSize = 14.sp)
                    Spacer(Modifier.height(16.dp))
                    OutlinedButton(onClick = vm::refreshHistory) { Text("重试") }
                }
                is HistoryState.Loaded -> if (records.isEmpty()) {
                    Column(
                        Modifier.fillMaxSize().padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        Icon(Icons.Filled.History, contentDescription = null, tint = HistoryGreen, modifier = Modifier.size(52.dp))
                        Spacer(Modifier.height(18.dp))
                        Text("还没有识别记录", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = HistoryInk)
                        Spacer(Modifier.height(6.dp))
                        Text("拍一张叶子，记录会保存在这里", fontSize = 14.sp, color = HistoryMuted)
                        Spacer(Modifier.height(18.dp))
                        OutlinedButton(onClick = onGoHome) { Text("去拍一张") }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(1.dp),
                    ) {
                        items(records, key = HistoryRecord::id) { record ->
                            HistoryRow(record, onClick = { onOpenDetail(record.id) })
                        }
                        if (records.size < total) {
                            item {
                                OutlinedButton(
                                    onClick = vm::loadMoreHistory,
                                    enabled = !loadingMore,
                                    modifier = Modifier.fillMaxWidth().padding(16.dp),
                                ) {
                                    if (loadingMore) CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                                    else Text("加载更多")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryRow(record: HistoryRecord, onClick: () -> Unit) {
    val top = record.top
    Surface(color = Color.White, modifier = Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            HistoryThumbnail(record.imagePath)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(top?.labelCn ?: "未识别", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = HistoryInk)
                Text(
                    formatRecordDate(record.createdAt) + (record.feedback?.let { if (it.isCorrect) " · 已确认" else " · 已纠错" } ?: ""),
                    fontSize = 13.sp,
                    color = HistoryMuted,
                )
            }
            Spacer(Modifier.width(8.dp))
            Text("${((top?.prob ?: 0.0) * 100).toInt()}%", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = HistoryGreen)
        }
    }
}

@Composable
private fun HistoryThumbnail(path: String?) {
    val bitmap = path?.let { remember(it) { runCatching { BitmapFactory.decodeFile(it) }.getOrNull() } }
    Box(
        Modifier.size(56.dp).clip(RoundedCornerShape(6.dp)).background(Color(0xFFE3EAE2)),
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap != null) {
            Image(bitmap.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        } else {
            Icon(Icons.Filled.History, contentDescription = null, tint = HistoryGreen)
        }
    }
}

internal fun formatRecordDate(value: String): String = runCatching {
    OffsetDateTime.parse(value).format(DateTimeFormatter.ofPattern("MM-dd HH:mm"))
}.getOrDefault(value.replace('T', ' ').take(16))
