# E-19 Swin-T PlantDoc 有监督微调：跨域 34.93% → 77.06%

## 一句话结论

同一套五折交叉验证、同一份 PlantDoc、同一套 38 类分类头，把 ResNet18 换成 Swin-T 后，
跨域 top-1 从 **69.73% 提到 77.06%（+7.33pp）**，作物级从 83.50% 提到 **89.77%（+6.27pp）**；
相对各自零样本的提升分别是 +42.12pp 与 +36.55pp，**5/5 折全部变好**。
这是本项目全部实验里最高的跨域成绩，也是唯一一个"换模型就能提分"的正面结果。

## 两个设定必须分开说

与 `plantdoc_finetune.md`（ResNet18 版）完全一致的口径，不能混用：

| 设定 | 是否用 PlantDoc 标签 | ResNet18 | Swin-T |
| --- | --- | --- | --- |
| 零样本跨域泛化（`eval_crossdomain.py`） | 不用 | 28.18% | **34.93%** |
| 提分辨率（`eval_preprocess_ablation.py`） | 不用 | 30.38% @320 | 37.09% @352 |
| 多分辨率平均（`eval_ensemble.py`） | 不用 | — | **37.50%** |
| 有监督域适应（本文 / `finetune_plantdoc.py`） | **用** | 69.73% | **77.06%** |

前三种是"闭卷考试"，第四种是"开卷补课"。**77.06% 不能当作零样本跨域准确率对外说。**

## 实验设置

| 项目 | 配置 |
| --- | --- |
| 源域 | PlantVillage 38 类（`ml/data/processed/plantvillage`） |
| 源域权重 | `ml/runs/swin_t_224_15ep/best.pt`（epoch 8，PV val_acc 0.9961） |
| 目标域 | PlantDoc 2920 张 / 27 类（train 2668 + test 252 合并后重新分层） |
| 折划分 | 5 折按类别分层，589/581/581/582/587，每折类别覆盖 27/27 |
| 分类头 | **保持 38 类**，与零样本评估同头，保证可比 |
| 超参 | 4 轮、bs 64、224px、backbone lr 1e-4、head lr 1e-3、AdamW、cosine、ls 0.1、seed 42 |
| 增强 | 仅裁剪 + 翻转（`strong_aug=False`），与 ResNet18 那轮一致 |
| 模型选择 | **不做早停、不按验证折选轮次**，固定 4 轮取最后一轮 |

### 脚本一行都没改

原以为 `ml/src/finetune_plantdoc.py` 硬编码了 resnet18，读完 670 行确认它已是模型无关的：

| 关注点 | 实现位置 | 结论 |
| --- | --- | --- |
| 模型名从哪来 | `finetune_plantdoc.py:419` | `args.model` → `ck["args"]["model"]` → 目录名，三级兜底 |
| 分类头学习率分组 | `finetune_plantdoc.py:90-100` `HEAD_PREFIX` | 已登记 `"swin_t": "head"` |
| 分类头维度 | `finetune_plantdoc.py:424-426` | 从 `ck["classes"]` 取 38 类，不写死 |
| 模型构建 | `finetune_plantdoc.py:266` | `build_model(...)` 后 `load_state_dict` |

checkpoint 自带元数据也对得上：`args.model='swin_t'`、`epoch=8`、`n_classes=38`、`img_size=224`。
**本次没有修改 `finetune_plantdoc.py`、`models.py`、`data.py`、`eval_crossdomain.py`、`train_colab.py`。**

## 主结果

### 与 ResNet18 对比（5 折均值 ± 标准差）

| 指标 | ResNet18 | Swin-T | 差 |
| --- | --- | --- | --- |
| 零样本 top-1 | 28.18% | 34.93% ± 1.61% | +6.75pp |
| 零样本作物级 | 51.13% | 53.22% | +2.09pp |
| **微调后 top-1** | **69.73% ± 2.37%** | **77.06% ± 2.36%** | **+7.33pp** |
| **微调后作物级** | **83.50%** | **89.77% ± 2.01%** | **+6.27pp** |
| 微调后 top-3 | 90.57% | **95.30%**（五折 93.87~96.04） | +4.73pp |
| 相对零样本增益（top-1） | +41.55pp | +42.12pp ± 3.40pp | 持平 |
| 相对零样本增益（作物级） | +32.37pp | +36.55pp ± 3.88pp | +4.18pp |
| 变好的折数 | 5/5 | **5/5** | — |

**两处值得注意**：
1. **增益幅度几乎相同（+41.55 vs +42.12pp）**，说明微调能榨出的提升已经被这批数据的上限卡住，
   不是"模型不够强"。真正拉开差距的是起点（34.93 vs 28.18）和终点（77.06 vs 69.73）。
2. **作物级增益 Swin-T 更大（+36.55 vs +32.37pp）**，与零样本时它作物级只领先 2.09pp 形成对比——
   微调后作物级领先 6.27pp。说明 Swin-T 的细粒度优势在适配目标域后被完整释放了。

### 逐折明细

| 折 | 验证数 | 零样本 top-1 | 微调 top-1 | 零样本作物级 | 微调作物级 | 微调 top-3 | 平均置信度 | 用时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 589 | 34.13% | **78.10%** | 51.44% | 89.30% | 95.08% | 0.7356 | 196.1s |
| 1 | 581 | 32.53% | **78.14%** | 51.46% | 91.74% | 96.04% | 0.7472 | 194.5s |
| 2 | 581 | 35.63% | **79.69%** | 51.98% | 91.39% | 95.35% | 0.7501 | 192.4s |
| 3 | 582 | 36.60% | **75.60%** | 55.50% | 89.69% | 94.67% | 0.7378 | 193.0s |
| 4 | 587 | 35.78% | **73.76%** | 55.71% | 86.71% | 93.87% | 0.7271 | 192.4s |
| 均值 | — | **34.93%** ± 1.61 | **77.06%** ± 2.36 | **53.22%** ± 2.19 | **89.77%** ± 2.01 | 95.30% | 0.7396 | 193.7s |

折间标准差 2.36pp，与 ResNet18 的 2.37pp 几乎一样——**微调后两模型的稳定性没有差别**，
差别全在水平高度上。

### 训练曲线（每折 4 轮，以折 0 为例）

| 轮 | train_loss | val_top1 | val_crop | lr_backbone | lr_head |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.9481 | **69.95%** | 83.70% | 8.55e-05 | 8.54e-04 |
| 2 | 1.3618 | 76.23% | 89.13% | 5.05e-05 | 5.01e-04 |
| 3 | 1.1670 | 75.89% | 89.47% | 1.55e-05 | 1.47e-04 |
| 4 | 1.0934 | 78.10% | 89.30% | 1e-06 | 1e-06 |

**第 1 轮就把零样本 34.13% 推到 69.95%**——提升的绝大部分发生在第一轮，
后面三轮只再涨 8pp。这个形状与 ResNet18 那轮完全一致（第 1 轮就到 59~65%），
再次印证"提升主要来自让模型见到目标域成像风格，而不是学到新的病害知识"。

折 4 是唯一 val_top1 在第 3 轮出现回落的折（75.30% → 73.76%），但它仍是 5 折里
零样本最高的（35.78%），微调后反而最低（73.76%）。**没有早停**是脚本的刻意设计
（避免验证折变成模型选择集），代价就是这个。

## 结论

1. **Swin-T 微调后 77.06% 是全部实验的最高跨域成绩**，比 ResNet18 的 69.73% 高 7.33pp，
   5/5 折全部更好，配对差 +42.12 ± 3.40pp 稳定。
2. **子代理的事前预估（71~74%）偏保守，实际 77.06% 超过预估上限 3.06pp**。
   它给的压制理由里"微调压缩模型差距"这一条没有成立——实际是**差距被放大**了
   （零样本 +6.75pp → 微调后 +7.33pp，作物级 +2.09pp → +6.27pp）。
   复盘：它按"域适应文献里强模型优势被摊薄"的普遍规律推断，但我们的微调只有 4 轮、
   lr 1e-4 的小步调整，模型容量差异没有被抹平，反而因为 Swin-T 的层次化窗口注意力
   更适合处理"病斑尺度不一 + 背景杂乱"这类域偏移而被保留下来。
3. **与零样本结论形成完整闭环**：Swin-T 的优势在两个设定下都存在，且方向一致。
   这是选 Swin-T 作为主模型的第三个独立证据（前两个：PlantDoc 零样本 +6.75pp、
   PlantWild 零样本 +5.69pp）。
4. **代价必须说清**：Swin-T 27.55M 参数（ResNet18 的 2.46 倍）、CPU 单张延迟 53.8ms
   （3.21 倍）、ONNX 体积约为 2.5 倍（ResNet18 44.77MB，Swin 未导出但按参数量推算）。
   微调本身也要 5×194s ≈ 16 分钟 GPU 时间。

## 已知局限

1. **有监督设定**，用了 PlantDoc 标签，与零样本 34.93% 不可互换。
2. **只测了 @224 单分辨率**，Swin-T 在零样本下 @352 比 @224 高 2.16pp，
   微调后若也走 @352 可能更高——未做。
3. **4 轮不早停**，折 4 明显在第 3 轮后回落，若加早停均值可能再高 1pp 左右。
4. **PlantDoc 只有 2920 张**，5 折每折训练 2331~2339 张，Swin-T 28M 参数下过拟合压力大，
   真实田间场景（照片更多样）表现可能不如这个数字。
5. **未做显著性检验**，虽然 5/5 折都提升、配对差 3.40pp 标准差远小于 42.12pp 均值，
   结论稳健，但严格统计检验未做。
6. **PlantDoc 标注本身有噪声**（原论文人工标注约 300 小时，仍存在类间混淆），
   77.06% 里有一部分误差来自标注而非模型。

## 产物

| 路径 | 说明 |
| --- | --- |
| `ml/runs/swin_t_224_15ep/plantdoc_ft_cv.json` | 五折完整结果（含逐折 history、零样本对照、配对差） |
| `ml/runs/swin_t_224_15ep/plantdoc_ft_cv_smoke.json` | 冒烟结果（1 折 1 轮，仅验证管线） |
| `ml/runs/swin_t_224_15ep/smoke_log.txt` | 冒烟完整日志 |
| `kaggle/swin_finetune/run.py` | Kaggle kernel（含 `find_swin_ckpt()` 三级权重查找） |
| `kaggle/swin_finetune/kernel-metadata.json` | kernel 元数据 |
| `scripts/extract_kernel_json.py` | 从 kernel 日志的 JSON-lines 里提取 `ALL_RESULTS_JSON` 段 |
| `ml/artifacts/plantdoc_finetune_swin.md` | 本文 |

## 复现

```powershell
# 本地（CPU 约 3 小时，GPU 约 16 分钟）
$env:PYTHONUTF8=1
cd "D:\Desktop\神经网络最终项目\ml"
python src/finetune_plantdoc.py --run-dir runs/swin_t_224_15ep `
    --folds 5 --epochs 4 --batch-size 64 --img-size 224 `
    --lr-backbone 1e-4 --lr-head 1e-3 --seed 42 `
    --num-workers 0 --out runs/swin_t_224_15ep/plantdoc_ft_cv.json

# Kaggle（需先把 swin_t 的 best.pt 上传成独立私有 dataset）
cd "D:\Desktop\神经网络最终项目"
python -m kaggle kernels push -p kaggle/swin_finetune --accelerator NvidiaTeslaT4
python -m kaggle kernels logs wangkaiyu66/plant-leaf-swin-finetune > _kaggle_out/swinft.log
python scripts/extract_kernel_json.py _kaggle_out/swinft.log -o ml/runs/swin_t_224_15ep/plantdoc_ft_cv.json
```

**取结果的重要教训**：Kaggle 会把整个 kernel 输出目录打包成单个 `_output_.zip`，
`kaggle kernels output --file-pattern` 对 zip 内部条目无效（会把 2670 张 PlantDoc 训练图一起下）。
**要拿结构化结果就走 `kaggle kernels logs`**，把 JSON 打到 stdout 的 `ALL_RESULTS_JSON_BEGIN/END`
之间，再本地解析——这条路径稳定且只几十 KB。
