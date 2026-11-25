package com.wkylnj.plantdoctor.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.wkylnj.plantdoctor.data.Health

private val AboutGreen = Color(0xFF246B4A)
private val AboutInk = Color(0xFF19231D)
private val AboutMuted = Color(0xFF59665E)
private val AboutPaper = Color(0xFFF5F7F2)
private val AboutWarn = Color(0xFF8C4A0F)

@Composable
fun AboutScreen(vm: MainViewModel, onBack: () -> Unit) {
    val health by vm.health.collectAsStateWithLifecycle()
    Surface(color = AboutPaper, modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                Modifier.fillMaxWidth().height(60.dp).padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = AboutInk) }
                Text("关于", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = AboutInk)
            }
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                Column(Modifier.fillMaxWidth().padding(vertical = 16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("叶片医生", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = AboutGreen)
                    Spacer(Modifier.height(4.dp))
                    Text("版本 0.1.0", fontSize = 14.sp, color = AboutMuted)
                }

                SectionTitle("当前识别模型")
                InfoSection {
                    InfoLine("服务状态", if (health == null) "暂不可连接" else "在线")
                    InfoLine("模型结构", health?.architecture ?: "未连接")
                    InfoLine("推理后端", health?.model ?: "未连接")
                    InfoLine("模型版本", health?.version ?: "—")
                    InfoLine("可识别类别", health?.let { "${it.classes} 种" } ?: "—")
                    InfoLine("输入尺寸", health?.let { "${it.imageSize} × ${it.imageSize}" } ?: "224 × 224")
                }

                SectionTitle("训练数据")
                InfoSection {
                    Text("PlantVillage 公开叶片图像数据集", fontSize = 15.sp, color = AboutInk)
                    Spacer(Modifier.height(5.dp))
                    Text("包含多种作物与病害类别，图像主要来自实验室拍摄环境。", fontSize = 14.sp, color = AboutMuted, lineHeight = 20.sp)
                }

                SectionTitle("使用说明")
                Surface(color = Color(0xFFFFF2DE), shape = RoundedCornerShape(8.dp)) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("结果仅供参考", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = AboutWarn)
                        Text(
                            "模型在实验室图像上训练和评估，田间光照、背景与拍摄角度不同，识别结果可能明显变化。不能据此自行用药，也不能替代专业植保诊断。",
                            fontSize = 14.sp,
                            color = AboutInk,
                            lineHeight = 21.sp,
                        )
                    }
                }

                InfoSection {
                    InfoLine("隐私", "照片保存在本机；清空记录时一并删除")
                    InfoLine("反馈", "仅提交识别结果和匿名记录编号")
                }
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(text, fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = AboutInk)
}

@Composable
private fun InfoSection(content: @Composable () -> Unit) {
    Surface(color = Color.White, shape = RoundedCornerShape(8.dp)) {
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp), content = content)
    }
}

@Composable
private fun InfoLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Top) {
        Text(label, color = AboutMuted, fontSize = 14.sp)
        Spacer(Modifier.width(12.dp))
        Text(value, color = AboutInk, fontSize = 14.sp, fontWeight = FontWeight.Medium)
    }
}
