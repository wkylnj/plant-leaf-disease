package com.wkylnj.plantdoctor

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.content.Intent
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.wkylnj.plantdoctor.data.UiState
import com.wkylnj.plantdoctor.data.Screen
import com.wkylnj.plantdoctor.ui.AboutScreen
import com.wkylnj.plantdoctor.ui.DetailScreen
import com.wkylnj.plantdoctor.ui.FeedbackSheet
import com.wkylnj.plantdoctor.ui.HistoryScreen
import com.wkylnj.plantdoctor.ui.MainScreen
import com.wkylnj.plantdoctor.ui.MainViewModel
import java.io.File

/**
 * 唯一的 Activity。
 *
 * 拍照用系统相机 Intent 而不是 CameraX：
 * CameraX 要处理预览、对焦、旋转、权限四套 API，Alpha 阶段是风险最高的部分
 * （docs/实验指导手册.md 里把"卡超 2 天"列为降级触发条件）。
 * 系统相机一行 Intent 就能拿到一张全尺寸照片，画质还更好。
 */
class MainActivity : ComponentActivity() {

    private var pendingPhoto: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val vm: MainViewModel = viewModel()
            var screen by remember { mutableStateOf<Screen>(Screen.Home) }
            var feedbackRequestId by remember { mutableStateOf<String?>(null) }
            var showPermissionHelp by remember { mutableStateOf(false) }

            BackHandler(enabled = screen != Screen.Home || feedbackRequestId != null) {
                if (feedbackRequestId != null) feedbackRequestId = null
                else screen = (screen as? Screen.Detail)?.from ?: Screen.Home
            }

            // 相册选择
            val galleryLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.GetContent()
            ) { uri: Uri? ->
                if (uri != null) {
                    val f = copyToCache(uri)
                    if (f != null) {
                        vm.recognize(f)
                    }
                }
            }

            // 系统相机
            val cameraLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.TakePicture()
            ) { ok ->
                val f = pendingPhoto
                if (ok && f != null && f.exists() && f.length() > 0) {
                    vm.recognize(f)
                }
            }

            val permLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestPermission()
            ) { granted ->
                if (granted) launchCamera(cameraLauncher)
                else showPermissionHelp = true
            }

            LaunchedEffect(Unit) { vm.checkHealth() }
            LaunchedEffect(screen) {
                when (val current = screen) {
                    Screen.History -> vm.refreshHistory()
                    is Screen.Detail -> vm.loadRecord(current.recordId)
                    else -> Unit
                }
            }

            when (val current = screen) {
                Screen.Home -> MainScreen(
                    vm = vm,
                    onPickCamera = {
                        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                        if (granted) launchCamera(cameraLauncher) else permLauncher.launch(Manifest.permission.CAMERA)
                    },
                    onPickGallery = { galleryLauncher.launch("image/*") },
                    onOpenHistory = { screen = Screen.History },
                    onOpenAbout = { screen = Screen.About },
                    onOpenDetail = { id -> screen = Screen.Detail(id, Screen.Home) },
                    onCorrect = { id -> vm.beginFeedback(); feedbackRequestId = id },
                )
                Screen.History -> HistoryScreen(
                    vm = vm,
                    onBack = { screen = Screen.Home },
                    onOpenDetail = { id -> screen = Screen.Detail(id, Screen.History) },
                    onGoHome = { screen = Screen.Home },
                )
                is Screen.Detail -> DetailScreen(
                    vm = vm,
                    onBack = { screen = current.from },
                    onCorrect = { id -> vm.beginFeedback(); feedbackRequestId = id },
                )
                Screen.About -> AboutScreen(vm = vm, onBack = { screen = Screen.Home })
            }

            feedbackRequestId?.let { id ->
                FeedbackSheet(vm = vm, requestId = id, onDismiss = { feedbackRequestId = null })
            }

            if (showPermissionHelp) {
                AlertDialog(
                    onDismissRequest = { showPermissionHelp = false },
                    title = { Text("需要相机权限") },
                    text = { Text("允许相机权限后才能直接拍照。你也可以从相册选择已有照片。") },
                    confirmButton = {
                        TextButton(onClick = {
                            showPermissionHelp = false
                            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)))
                        }) { Text("去设置") }
                    },
                    dismissButton = {
                        TextButton(onClick = { showPermissionHelp = false; galleryLauncher.launch("image/*") }) { Text("从相册选择") }
                    },
                )
            }
        }
    }

    private fun launchCamera(launcher: androidx.activity.result.ActivityResultLauncher<Uri>) {
        val dir = File(cacheDir, "captures").apply { mkdirs() }
        val f = File(dir, "leaf_${System.currentTimeMillis()}.jpg")
        pendingPhoto = f
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", f)
        launcher.launch(uri)
    }

    /** 相册返回的 Uri 不能直接读文件路径，先拷到 cache 再上传。 */
    private fun copyToCache(uri: Uri): File? = runCatching {
        val dir = File(cacheDir, "captures").apply { mkdirs() }
        val f = File(dir, "picked_${System.currentTimeMillis()}.jpg")
        contentResolver.openInputStream(uri)?.use { input ->
            f.outputStream().use { out -> input.copyTo(out) }
        }
        f.takeIf { it.length() > 0 }
    }.getOrNull()
}
