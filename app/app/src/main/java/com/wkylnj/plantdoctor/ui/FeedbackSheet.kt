package com.wkylnj.plantdoctor.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.wkylnj.plantdoctor.data.DiseaseClass

private enum class FeedbackStage { Start, Crop, Disease }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeedbackSheet(vm: MainViewModel, requestId: String, onDismiss: () -> Unit) {
    val classes by vm.classes.collectAsStateWithLifecycle()
    val doneIds by vm.feedbackDoneIds.collectAsStateWithLifecycle()
    val busy by vm.feedbackBusy.collectAsStateWithLifecycle()
    val error by vm.feedbackError.collectAsStateWithLifecycle()
    var stage by remember(requestId) { mutableStateOf(FeedbackStage.Start) }
    var crop by remember(requestId) { mutableStateOf<String?>(null) }
    LaunchedEffect(requestId) { vm.beginFeedback() }

    ModalBottomSheet(onDismissRequest = { if (!busy) onDismiss() }, containerColor = Color(0xFFF5F7F2)) {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 20.dp).padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (requestId in doneIds) {
                Text("感谢你的反馈", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Color(0xFF174A35))
                Text("这条记录已更新。", fontSize = 15.sp, color = Color(0xFF59665E))
                Button(onClick = onDismiss, modifier = Modifier.fillMaxWidth().height(50.dp), shape = RoundedCornerShape(8.dp)) {
                    Text("完成")
                }
            } else {
                when (stage) {
                    FeedbackStage.Start -> {
                        Text("识别结果不对？", fontSize = 21.sp, fontWeight = FontWeight.Bold, color = Color(0xFF19231D))
                        Text("告诉我们正确答案，可以帮助改进识别。", fontSize = 14.sp, color = Color(0xFF59665E))
                        Spacer(Modifier.height(4.dp))
                        Button(
                            onClick = { stage = FeedbackStage.Crop },
                            modifier = Modifier.fillMaxWidth().height(50.dp),
                            shape = RoundedCornerShape(8.dp),
                        ) { Text("我知道正确的病害") }
                        OutlinedButton(
                            enabled = !busy,
                            onClick = { vm.sendFeedback(requestId, false) },
                            modifier = Modifier.fillMaxWidth().height(50.dp),
                            shape = RoundedCornerShape(8.dp),
                        ) { Text("我也不确定，只记录识别错误") }
                    }
                    FeedbackStage.Crop -> {
                        TextButtonRow("返回", onClick = { stage = FeedbackStage.Start })
                        Text("这是哪种作物？", fontSize = 21.sp, fontWeight = FontWeight.Bold, color = Color(0xFF19231D))
                        Text("先选择作物，再选择具体病害。", fontSize = 14.sp, color = Color(0xFF59665E))
                        val crops = classes.map(DiseaseClass::cropCn).distinct().filter { it.isNotBlank() }.sorted()
                        if (crops.isEmpty()) {
                            Text("病害目录暂不可用，请稍后重试。", color = Color(0xFF9F3030), fontSize = 14.sp)
                        } else {
                            LazyVerticalGrid(
                                columns = GridCells.Fixed(2),
                                modifier = Modifier.fillMaxWidth().heightIn(max = 390.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                items(crops) { item ->
                                    OutlinedButton(
                                        onClick = { crop = item; stage = FeedbackStage.Disease },
                                        modifier = Modifier.fillMaxWidth().height(52.dp),
                                        shape = RoundedCornerShape(8.dp),
                                    ) { Text(item) }
                                }
                            }
                        }
                    }
                    FeedbackStage.Disease -> {
                        TextButtonRow("$crop · 更换作物", onClick = { stage = FeedbackStage.Crop })
                        Text("具体是哪种病害？", fontSize = 21.sp, fontWeight = FontWeight.Bold, color = Color(0xFF19231D))
                        val diseases = classes.filter { it.cropCn == crop }
                        Column(
                            Modifier.fillMaxWidth().heightIn(max = 400.dp).verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            diseases.forEach { disease ->
                                OutlinedButton(
                                    enabled = !busy,
                                    onClick = { vm.sendFeedback(requestId, false, disease.label) },
                                    modifier = Modifier.fillMaxWidth().height(50.dp),
                                    shape = RoundedCornerShape(8.dp),
                                ) { Text(disease.labelCn, fontSize = 14.sp) }
                            }
                        }
                    }
                }
                error?.let { Text(it, color = Color(0xFF9F3030), fontSize = 14.sp) }
                if (busy) {
                    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color(0xFF246B4A), modifier = Modifier.height(28.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun TextButtonRow(label: String, onClick: () -> Unit) {
    androidx.compose.material3.TextButton(onClick = onClick) { Text(label, color = Color(0xFF246B4A)) }
}
