# Beta 第一轮：Swin-T 补训 + 域适应增强实验（Kaggle）

> 日期：2026-11-24 · 训练平台：Kaggle Tesla T4（免费额度）· 数据：PlantVillage 38 类（39153/4350/10878）

## 一句话结论

**域适应增强（da_aug）失败**：同域、跨域、作物级三项指标全部下降；**Swin-T 成为跨域最强模型**（34.93%，比 ResNet18 高 6.75pp），但代价是 6.5× 参数量。

## 实验背景

Alpha 冲刺结束时的问题：同域 test 99%+，但换 PlantDoc 真实田间照片跨域只有 25.8%~28.2%（E2 实验）。结合 E7 Grad-CAM 的发现（模型看的是整片叶子的全局统计特征，不是病斑局部形态），本轮做两件事：

1. **E-da_aug**：给 MobileNetV3 加一组针对成像条件的强增强，试图强迫模型学「成像无关」的特征；
2. **M-07 补做**：Swin-T 15 轮完整训练（Alpha 期间因 Colab 免费池 503 四次失败），补齐 Transformer 家族数据。

训练迁移到 Kaggle CLI（Colab 免费版单次约 100 分钟上限 + GPU 池长期无空位）。Kaggle：30h/周 GPU、12h/单次、`kernels push` 异步提交 + `kernels output` 取产物。两个 kernel 各跑一次成功：Swin-T 15 轮约 68 分钟、da_aug MobileNetV3 15 轮约 73 分钟，共耗 GPU 2.43h。

## 域适应增强的具体配置

`ml/src/train_colab.py` 的 `--da-aug` 开关，在 strong aug（RandAugment + ColorJitter(0.3,0.3,0.3,0.05) + RandomErasing）之上追加：

| 扰动 | 参数 | 针对什么 |
|---|---|---|
| ColorJitter | brightness/contrast/saturation 0.6, hue 0.15 | 田间光照/色温不稳 |
| RandomGrayscale | p=0.15 | 迫使模型不依赖颜色 |
| GaussianBlur | k=5, σ∈[0.1,2.0], p=0.3 | 对焦不清/运动模糊 |
| RandomAutocontrast | p=0.2 | 低对比度 |
| RandomEqualize | p=0.2 | 过曝/欠曝 |

## 完整结果

| 模型 | 同域 test | macro-F1 | 跨域 top-1 | 跨域 top-3 | 跨域作物级 | 置信度 | 参数量 |
|---|---|---|---|---|---|---|---|
| Swin-T (M-07) | **0.9962** | **0.9956** | **0.3493** | **0.5705** | 0.5322 | 0.5667 | 27.55M |
| MobileNetV3 基线 | 0.9948 | 0.9934 | 0.2582 | 0.4575 | 0.5195 | 0.4193 | 4.25M |
| MobileNetV3 + da_aug | 0.9909 | 0.9878 | 0.2360 | 0.4308 | 0.4990 | 0.4155 | 4.25M |
| ResNet18 | 0.9941 | 0.9916 | 0.2818 | 0.4901 | 0.5113 | 0.4366 | 11.20M |
| 双模型集成+TTA（Alpha 已测） | 0.9965 | — | 0.2911 | — | 0.5473 | — | 15.45M |

## 结论一：da_aug 失败，且失败方式很有信息量

三项指标全部下降：同域 −0.39pp、跨域 −2.22pp、作物级 −2.05pp。

**机制分析**：ColorJitter/灰度/模糊/自动对比度/均衡这些扰动改变的是**像素统计分布**。模型经过 15 轮训练后学会的是「忽略扰动后仍稳定的统计量」——但这些统计量仍然是 PlantVillage 同源数据的统计量。扰动没有改变模型**依赖的特征类型**（全局颜色/纹理统计），只是让它在这些统计量上做了一层不变性。换到 PlantDoc，统计量漂移依旧，跨域不升反降（增强还挤占了 15% 的有效训练信号，同域也掉了）。

这与 E7（Grad-CAM：注意力弥散、健康叶与病叶分布几乎相同）、E2（作物级 51% vs 病害级 28%：认得出作物认不出病）的机理判断一致，**三个独立实验指向同一根因：模型依赖全局统计特征而非局部病斑形态**。

**对路线的直接影响**：普通「调 transform」式增强这条路线证伪。要让模型被迫看局部，得动数据管线：
1. 局部裁剪训练（只喂叶片/病斑小块，多裁剪投票）
2. 背景替换（把实验室纯背景换成复杂背景，需分割或抠图）
3. PlantDoc 少量微调（域适应直接法）

## 结论二：Swin-T 跨域最强，但代价大

- 跨域 top-1 **34.93%**：比 ResNet18（28.18%）高 6.75pp、比 MobileNetV3（25.82%）高 9.11pp；
- 同域 99.62% 也是四模型最高；置信度 0.5667 明显更高（跨域时更「敢给」，但也更容易自信地错）；
- 代价：27.55M 参数（MobileNetV3 的 6.5×）、E3 实测 CPU 延迟 53.8ms（3.21×）。

猜测与 E7 结论吻合：Swin 的全局窗口注意力天然在做「整图区域间关系建模」，对局部像素统计漂移的敏感度低于 CNN 的卷积核；但它的「强」仍然建立在同源统计上，并没有真的去看病斑——所以也只是 35%，不是 70%。

## 错误模式（da_aug 模型，与基线几乎相同）

```
   129  21_Potato_Late_blight  →  30_Tomato_Late_blight     # 同一种病原
   110  35_Tomato_Yellow_Leaf_Curl_Virus → 30_Tomato_Late_blight
    85  09_Corn_maize_Northern_Leaf_Blight → 10_Corn_maize_healthy
    72  32_Tomato_Septoria_leaf_spot → 26_Strawberry_Leaf_scorch
```

健康类仍全军覆没（Cherry/Tomato/Peach healthy ≈0）：模型从没见过「脏的」健康叶片，把一切非标准外观都归为「有病」——这条 Alpha 期间的判断继续成立。

## 产物

| 路径 | 说明 |
|---|---|
| `ml/runs/swin_t_224_15ep/` | best.pt (105.4MB, epoch 8, val 0.9961) + history/curves/混淆矩阵/summary |
| `ml/runs/mobilenet_v3_large_daaug_224_15ep/` | best.pt (16.4MB, epoch 12, val 0.9940) + 同上 + crossdomain_preds.csv |
| `kaggle/swin_t/` `kaggle/da_aug_mobilenet/` `kaggle/smoke/` | kernel 源（run.py + metadata） |
| `scripts/kaggle_fetch.py` | 产物取回脚本（status/fetch） |

## 复现

```powershell
cd D:\Desktop\神经网络最终项目\ml
# 跨域评测（Swin-T）
python src\eval_crossdomain.py --run-dir runs\swin_t_224_15ep --plantdoc-root data\raw\plantdoc --split all
# 重新提交训练（Kaggle）
cd ..; python -m kaggle kernels push -p kaggle\swin_t --accelerator NvidiaTeslaT4
```

## 已知局限

1. da_aug 只测了一组超参（0.6/0.15/0.3/0.2 的组合是拍的），不排除更温和/更激进的组合有不同表现——但方向性结论（改分布≠改特征）不受影响；
2. Swin-T 只测了一个窗口配置（swin_t，window=7）；
3. 跨域评测映射 27 类（PlantDoc 剔除 1 类仅 2 张）；
4. 两模型置信度不可直接比（温度未校准）。
