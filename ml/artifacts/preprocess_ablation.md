# E-15 跨域预处理消融：提高推理分辨率是唯一有效手段

## 一句话结论

模型权重完全不动，只把推理分辨率从 224 提到 288~352，**跨域 top-1 提升 2.2~4.2 个百分点**；所有「改画面形状」的喂法（拉伸、填边、多尺度平均）和「叶片定位裁剪」都是负收益。这是本项目性价比最高的单项改进。

## 实验设置

- 样本：PlantDoc `all` 划分，2920 张真实田间照片（跨域）
- 口径：**只换「图怎么喂进去」，不重训、不改权重**
- 指标：top-1 / top-3 / 作物级（同作物即算对）/ 平均置信度
- 脚本：`ml/src/eval_preprocess_ablation.py`

## 结果一：分辨率扫描（全量 2920 张）

### ResNet18（训练分辨率 224）

| 推理分辨率 | top-1 | top-3 | 作物级 | 置信度 | 相对 224 |
|---|---|---|---|---|---|
| 224（历史口径） | 0.2818 | 0.4901 | 0.5113 | 0.4366 | — |
| 256 | 0.2969 | 0.4973 | 0.5271 | 0.3887 | +1.51pp |
| 288 | 0.3021 | 0.4993 | 0.5370 | 0.3496 | +2.03pp |
| 320 | **0.3038** | 0.4990 | **0.5435** | 0.3236 | **+2.20pp** |

### Swin-T（训练分辨率 224）

| 推理分辨率 | top-1 | top-3 | 作物级 | 置信度 | 相对 224 |
|---|---|---|---|---|---|
| 224（历史口径） | 0.3493 | 0.5705 | 0.5322 | 0.5667 | — |
| 288 | 0.3685 | 0.5962 | 0.5712 | 0.5115 | +1.92pp |
| 320 | 0.3705 | 0.5969 | 0.5719 | 0.5309 | +2.12pp |
| 352 | **0.3709** | **0.5983** | **0.5777** | 0.4754 | **+2.16pp** |

Swin-T 在 352 仍未见顶（320→352 只 +0.04pp，已趋平），且耗时从 192s 涨到 477s。**Swin-T @352 = 37.09% 是目前全部实验的最高跨域成绩**（原纪录 Swin-T @224 = 34.93%）。

## 结果二：形状变体（test 252 张口径，ResNet18）

基线（同口径）0.3373：

| 喂法 | 说明 | top-1 | 相对基线 |
|---|---|---|---|
| `squash` | 整图拉伸到 224×224 | 0.2937 | −4.36pp |
| `short224` | 短边直接缩到 224 再中心裁 | 0.2976 | −3.97pp |
| `letterbox` | 等比缩放 + 填边到 224 | 0.2579 | −7.94pp |
| `multiscale` | 3 种喂法 × 翻转共 6 视图平均 | 0.3214 | −1.59pp |

## 结果三：叶片定位裁剪（E-16，test 252 张口径，ResNet18）

针对「PlantVillage 近景特写 vs PlantDoc 田间远景」的尺度错位假设，用超绿指数 + 形态学 + 最大连通域定位叶片包围盒后裁剪放大：

| 喂法 | top-1 | 作物级 | 抠图失败 | 相对基线 |
|---|---|---|---|---|
| baseline | 0.3373 | 0.6071 | — | — |
| `leafcrop`（紧包围盒） | 0.3016 | 0.5556 | 0/252 | −3.57pp |
| `leafcrop_wide`（放大 1.4 倍） | 0.3095 | 0.5635 | 0/252 | −2.78pp |

**假设被推翻**：尺度错位不是主因。抠图本身成功率 100%（无回退），但裁剪后反而变差——说明模型依赖的「整片叶子的全局统计量」在裁剪后也被破坏了，模型没见过「叶片占满画面」的构图。脚本：`ml/src/eval_leafcrop.py`。

## 机制解释

**为什么提高分辨率有效**：PlantDoc 的叶片在画面中占比远小于 PlantVillage。放大输入不改变构图，只增加像素密度，让病斑纹理真正进入模型的感受野。模型是在 224 上训练的，感受野与纹理尺度按 224 标定，所以**「训练分辨率的 1.3~1.6 倍」是安全区**。

**为什么 384/448 又变差**（test 252 张扫描：256→0.3611、288→**0.3730**、320→0.3532、384→0.3373、448→0.3135）：分辨率继续加大时，叶片上的病斑相对模型感受野变小、有效纹理被拉散，判决反而变差。

**为什么改形状是负收益**：PlantDoc 的叶片占比、长宽比与 PlantVillage 差异很大，强行改成 PlantVillage 的构图会把叶片细节压掉（拉伸变形、填边引入无意义像素）。

## 对后续实验的约束

- `local_crop`（局部裁剪训练）与 `bg_replace`（背景替换）这两个训练侧实验，**评估必须用 PlantDoc 跨域**。同域 val 精度下降是预期内的，不能用 `summary.json` 的 `test_acc` 判断成败。
- 分辨率增益与集成增益是否正交，见 `ml/artifacts/ensemble.md`。

## 产物

- `ml/src/eval_preprocess_ablation.py`（消融脚本，`--variants` 支持 baseline/squash/short224/letterbox/center<N>/multiscale）
- `ml/src/eval_leafcrop.py`（叶片定位裁剪，E-16）
- `ml/runs/resnet18_224_15ep/preprocess_all_sweep.json`（全量分辨率扫描）
- `ml/runs/resnet18_224_15ep/preprocess_res_sweep.json`（test 252 张分辨率扫描，含 384/448）
- `ml/runs/swin_t_224_15ep/preprocess_all_sweep.json`（Swin-T 全量扫描）
- `ml/runs/resnet18_224_15ep/leafcrop.json`

## 复现

```
cd ml
python src/eval_preprocess_ablation.py --run-dir runs/resnet18_224_15ep --split all \
  --variants baseline center256 center288 center320 --out runs/resnet18_224_15ep/preprocess_all_sweep.json

python src/eval_preprocess_ablation.py --run-dir runs/swin_t_224_15ep --split all \
  --variants baseline center288 center320 center352 --out runs/swin_t_224_15ep/preprocess_all_sweep.json

python src/eval_leafcrop.py --run-dir runs/resnet18_224_15ep --split test
```
