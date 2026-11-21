# ONNX 导出实测

## 为什么导出 ONNX

训练产物是 `.pt`（PyTorch checkpoint），只能在装有 PyTorch 的环境里加载。服务端若要部署，PyTorch 运行时约 800 MB，而 ONNX Runtime 只要约 15 MB，且推理更快、跨语言（Python / Java / C++ 均可调用）。安卓端接推理时，ONNX 也是比 PyTorch Mobile 更通用的选择。

## 之前为什么失败

训练脚本 `ml/src/train_colab.py:663-697` 里有完整导出逻辑，但 Colab 环境缺少 `onnxscript` 模块，报错被外层 `except Exception as e` 吞掉，只留下一行：

```
ONNX 导出失败（不影响训练结果）: No module named 'onnxscript'
```

`onnxscript` 是 torch >= 2.5 的 **dynamo 导出器**必需的依赖。旧版 TorchScript 导出器不需要它，但新版默认走 dynamo。

## 解决方式

不再依赖训练时顺手导出，改为独立脚本 `ml/src/export_onnx.py`，对任意已有 checkpoint 补做导出：

```bash
cd ml
python src/export_onnx.py --run-dir runs/resnet18_224_15ep
python src/export_onnx.py --run-dir runs/mobilenet_v3_large_224_15ep
```

环境依赖：`pip install onnxscript onnx onnxruntime`

脚本做了三件事：

1. 从 `best.pt` 读回模型名、类别数、权重，重建模型结构
2. 导出为 opset 17，输入名 `input`、输出名 `logits`，batch 维设为动态（`{"input": {0: "batch"}}`），这样服务端可一次推理多张
3. 用 onnxruntime 跑同一批输入，与 PyTorch 输出比对

## 实测结果

| 模型 | 参数量 | ONNX 大小 | 导出耗时 | 最大绝对误差 | 分类结果一致 |
|---|---:|---:|---:|---:|---|
| ResNet18 | 11.196 M | **44.77 MB** | 1.12 s | 1.37e-06 | ✅ |
| MobileNetV3-Large | 4.251 M | **16.99 MB** | 0.74 s | 3.58e-06 | ✅ |

误差量级在 float32 累加误差的正常范围内（1e-6），argmax 预测完全一致——说明导出没有损失精度，可以直接部署。

## 一个顺带得到的结论

**MobileNetV3 的 ONNX 只有 ResNet18 的 38%（16.99 MB vs 44.77 MB）**，这与参数量比例（4.251 M / 11.196 M = 38%）以及前面的精度结果（0.9948 vs 0.9941，MobileNetV3 反而略高）完全吻合。

三组独立数据指向同一结论：**在这个任务上，轻量模型在「精度 / 参数量 / 部署体积」三个维度同时占优**。这是报告里「手机端选型」一节最有力的证据，不只是「轻量模型稍差但能接受」，而是「轻量模型更好且更小」。

## 已知限制

1. 只验证了 ResNet18 与 MobileNetV3，其余四个模型（resnet50 / shufflenet / efficientnet / swin_t）没有 checkpoint，未导出。
2. 动态 batch 只设了第 0 维，图像尺寸固定 224×224。若服务端要接受任意尺寸，需在预处理阶段 resize。
3. 未做 INT8 量化（X-04 仍待办）。量化后体积可再降到约 1/4，但会引入少量精度损失，需单独实验验证。
4. 未在安卓端实测 onnxruntime-mobile 推理速度，目前安卓端仍走服务端 API。

## 产物

- `ml/src/export_onnx.py` — 导出与校验脚本
- `ml/runs/resnet18_224_15ep/model.onnx`（44.77 MB）、`onnx_meta.json`
- `ml/runs/mobilenet_v3_large_224_15ep/model.onnx`（16.99 MB）、`onnx_meta.json`

> 注：`*.onnx` 已加入 `.gitignore`（单文件 17–45 MB，不进仓库），只提交 `onnx_meta.json` 作为导出记录。需要模型文件时在本地重跑上面的命令即可，约 1 秒。