package com.wkylnj.plantdoctor

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.LaunchedEffect
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.wkylnj.plantdoctor.data.UiState
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
            }

            LaunchedEffect(Unit) { vm.checkHealth() }

            MainScreen(
                vm = vm,
                onPickCamera = {
                    val granted = ContextCompat.checkSelfPermission(
                        this, Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
                    if (granted) launchCamera(cameraLauncher)
                    else permLauncher.launch(Manifest.permission.CAMERA)
                },
                onPickGallery = { galleryLauncher.launch("image/*") },
            )
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
