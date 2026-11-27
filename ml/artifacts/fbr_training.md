# FBR 背景重组训练：文献里 +9.9pp 的方法，在我们这里是 −9.52pp

## 一句话结论

**FBR（Field Background Recomposition）在我们这个任务上失败了：跨域 top-1 从 33.73% 掉到 24.21%（−9.52pp），同域也从 99.41% 掉到 94.39%（−5.02pp）。** 这是第七条被证伪的路线，也是最重要的一条——它是文献里唯一报告了明确正收益的方法（Jeon et al., *Ecological Informatics* 2026，苹果叶上 +9.9pp）。

失败原因不是实现有 bug，而是**背景重组把训练分布改成了「病斑叶片贴在清晰田间背景上」，而这个组合在真实田间并不存在**：真实田间照片里，叶片本身就是模糊的、被遮挡的、与背景同色系的；我们的合成图把 PlantVillage 那种「高对比度、锐利、正对镜头的叶片」硬贴到背景上，反而制造了一种新的、更不自然的分布。

## 为什么值得单独做（而且值得单独写失败）

背景重组是这一轮最有希望的一条路线，理由有三：

1. **有明确的文献支撑**：FBR 论文在苹果叶真实环境上报告 baseline 47.1% → 57.0%（+9.9pp），叠加 CDAN+E 后到 91.1%。
2. **机制上讲得通**：`bg_causal.md` 已经用因果实验证明——只换背景就掉 16.00pp，同一批背景模糊后只掉 2.67pp。既然背景是域偏移的主要载体，那让模型在训练时就见到目标域背景，理应有效。
3. **它动了权重**：不像 TTA、颜色对齐那些只改输入的方法，FBR 真正改变了训练数据分布，是「让模型见到目标域成像风格」这一原则的直接实现。

所以它失败比它成功更有信息量：**说明「背景」这个因素虽然重要，但不能靠「贴上去」来解决。**

## 实验设置

| 项目 | 值 |
|---|---|
| 源域 | PlantVillage train 39170 张（38 类），全部重合成 |
| 合成方式 | `ml/src/make_fbr_dataset.py`：超绿+饱和度抠图 → 羽化 alpha → 随机缩放 55~85% / 旋转 ±25° / 贴到随机田间背景 |
| 背景库 | `ml/src/make_bg_library.py` 从 PlantDoc 筛出 **191 张**合格田间背景（拒绝 1751 张叶片过密、357 张白底、186 张水印、171 张含手指） |
| 训练集 | 39170 张合成图，0.1 切 val → train 35271 / val 3899 |
| 测试集 | **原始 PlantVillage test 10878 张**（不用合成图，否则测的是合成质量） |
| 模型 | ResNet18，15 轮，bs 64，lr 3e-4，与基线 `runs/resnet18_224_15ep` 完全一致 |
| 平台 | Kaggle T4（`plant-leaf-fbr` kernel） |

单变量对比：唯一改动是 `--data-root` 指向合成图目录，其余超参与基线一字不差。

## 主结果

| 指标 | 基线（PV 原图） | FBR 合成图 | 差 |
|---|---|---|---|
| 同域 PV test top-1 | 0.9941 | **0.9439** | **−5.02pp** |
| 同域 macro-F1 | 0.9916 | 0.9236 | −6.80pp |
| 跨域 PlantDoc test top-1（252 张） | 0.3373 | **0.2421** | **−9.52pp** |
| 跨域 top-3 | 0.6071 | 0.4960 | −11.11pp |
| 跨域作物级 | — | 0.4365 | — |
| 跨域平均置信度 | — | 0.4351 | — |

> 跨域用 `split=test`（252 张）而非全量 2920 张，是为了在 Kaggle 上快速出结果；基线同口径是 33.73%（见 `preprocess_ablation.md:53`），两者可比。1 张 = 0.40pp，所以 −9.52pp 相当于 24 张的差距，远超噪声。

## 四条结论

### 1. 同域掉 5pp 说明合成图本身「变难了」，但方向错了

模型在合成图上训练后，对原始 PV 测试集的准确率从 99.41% 掉到 94.39%。这说明合成图引入的扰动**没有被模型当成「无关变化」忽略掉**，而是挤占了一部分判别能力。

对照 `fbr_quality_diag.json`：单纯几何扰动（缩放+旋转）只掉 3.12pp，JPEG 压缩不掉，亮度增强反而 +0.12pp。所以这 5pp 主要来自**背景替换**本身，而不是几何/压缩伪影。

### 2. 跨域掉 9.52pp 说明「贴背景」不等于「见到目标域」

这是最关键的一条。按 `bg_causal.md` 的逻辑，模型依赖背景统计纹理，那把训练背景换成田间背景，模型就该学会忽略背景、转而依赖叶片本身。

实际结果相反。原因在于**合成图制造了一个真实田间不存在的组合**：

- 真实田间照片：叶片**也是**模糊/低对比/被遮挡/与背景同色系的（因为都在同一光照下、同一焦平面附近）
- 我们的合成图：叶片**锐利、高对比、正对镜头**（来自 PlantVillage），背景**模糊、低对比、复杂**（来自 PlantDoc）

模型于是学到一条捷径：**「叶片锐利 + 背景杂乱」这个组合 → 某类病害**。这条捷径在真实田间完全失效，因为真实照片里叶片和背景的清晰度是一致的。

换句话说，**FBR 引入了一个新的域偏移（清晰度不一致），抵消了它想解决的旧域偏移（背景类型不一致）。**

### 3. `10_Corn_maize_healthy` 成了新的「吸铁石」

FBR 模型的错误分布出现了明显的塌缩：191 个错误里 **73 个（38.2%）判给 `10_Corn_maize_healthy`**。

| 真实类别 | 被判成 Corn_healthy 的次数 |
|---|---|
| `09_Corn_Northern_Leaf_Blight` | 9 / 12 |
| `08_Corn_Common_rust` | 9 / 10 |
| `27_Strawberry_healthy` | 8 / 8 |
| `36_Tomato_mosaic_virus` | 7 / 10 |
| `20_Potato_Early_blight` | 5 / 14 |
| `30_Tomato_Late_blight` | 5 / 10 |
| `37_Tomato_healthy` | 4 / 8 |

`10_Corn_maize_healthy` 在 PlantVillage 里是「均匀黄绿色叶片 + 灰底」。合成之后，它的背景变成了各种田间场景，使它成为「任意绿色叶片 + 任意杂乱背景」的视觉均值——和 PlantWild 实验里 `19_Pepper_bell_healthy` 当吸铁石是同一个机制（见 `plantwild.md`）。

**这是一个可迁移的规律：只要某个类别在训练分布里覆盖了足够多的背景变体，它就会变成「兜底答案」。** FBR 恰好给了 `Corn_healthy` 最多的背景多样性，因为它的原始图最均匀、抠图最干净。

### 4. 与文献差异的原因

FBR 论文在苹果叶上 +9.9pp，我们在 27 类 13 物种上 −9.52pp。四个可能原因：

1. **类别数差距**：他们 4 类（苹果叶），我们 27 类跨 13 物种。「叶片锐利 + 背景杂乱」这条捷径在细粒度分类里危害更大——因为细粒度判别本来就更依赖纹理细节，而捷径会挤占这部分容量。
2. **抠图质量**：他们用 SAM（分割质量提升 87%），我们用超绿+Otsu。虽然掩膜覆盖率检查正常（mean 0.488，无一张 <0.15 或 >0.90），但边缘羽化和病斑保留仍不如 SAM。
3. **背景库规模与质量**：他们 200+ 种真实田间背景；我们只有 191 张，且目视约 4 成仍带水印或纯色布。
4. **他们的数据本身更难**：baseline 只有 47.1%（我们 99.41% 同域、33.73% 跨域），起点低时 UDA 增益空间更大——这与 `beta_round3_plan.md` §1.3 的判断一致。

## 对项目的意义

1. **七条证伪路线形成了完整闭环**：只改输入（TTA/颜色对齐/叶片裁剪）→ 失败；只改注意力范围（局部裁剪训练）→ 失败；改训练背景（FBR）→ 失败；**只有真正喂目标域真实图像（PlantDoc 微调 +41.55pp）才有效**。这把结论钉得更死：**域偏移不能靠合成来跨越，只能靠真实样本。**
2. **报告里这是很好的「预期落空」叙事**：有文献支撑、有机理支撑、有因果实验铺垫，结果仍然失败——正好呼应 `第11周心得.md` 里「AI 会把话说圆，但实验才作数」那一段。
3. **产品侧不受影响**：FBR 是我们自己造的训练数据，失败不影响现有交付（Swin-T 多分辨率平均 37.50% 仍是零样本最佳）。

## 已知局限

1. 只测了 ResNet18 单模型单分辨率，未在 Swin-T 上验证（但 Swin 微调成本高，且机制判断不依赖架构）。
2. 跨域只用了 PlantDoc test split（252 张），未跑全量 2920 张；1 张 = 0.40pp，但 −9.52pp 已远超噪声。
3. 背景库仅 191 张，且质量参差（约 4 成有水印或纯色布），可能放大了「新域偏移」。
4. 抠图用超绿+Otsu 而非 SAM，边缘质量不如文献。
5. 未做「只换背景不换叶片清晰度」的对照实验——这是验证「清晰度不一致」假设的直接方法，留待后续。

## 产物

| 文件 | 说明 |
|---|---|
| `ml/src/make_bg_library.py` | 背景库筛选（超绿掩膜 + 叶片占比/白底/水印/肤色过滤） |
| `ml/src/make_fbr_dataset.py` | FBR 合成（抠图 + 羽化 + 随机贴背景） |
| `ml/src/prepare_fbr_marker.py` | 合成图切 train/val + 挂原始 test + 写 `.prepared` |
| `kaggle/fbr/` | Kaggle kernel（训练） |
| `kaggle/eval_fbr/` | Kaggle kernel（评估，挂载训练输出直接跑） |
| `ml/artifacts/fbr_crossdomain_eval.json` | 跨域评估完整结果（含 per_class 与 top_confusions） |
| `ml/runs/resnet18_fbr_224_15ep/` | classes/config/summary/crossdomain.json |
| `ml/artifacts/fbr_quality_diag.json` | 单变量扰动分解（几何/压缩/亮度） |
| `ml/artifacts/bg_causal.md` | 背景因果分解（六种背景变体） |

## 复现

```bash
# 1. 抠背景库（约 10 分钟）
python ml/src/make_bg_library.py --plantdoc-root ml/data/raw/plantdoc --out ml/data/raw/plantdoc_bg --target 400

# 2. 合成（39170 张，约 20 分钟）
python ml/src/make_fbr_dataset.py --pv-root ml/data/processed/plantvillage/train \
    --bg-root ml/data/raw/plantdoc_bg --out ml/data/processed/pv_fbr --per-image 1 --size 224

# 3. 切 train/val 并挂原始 test
python ml/src/prepare_fbr_marker.py --fbr-root ml/data/processed/pv_fbr \
    --pv-root ml/data/processed/plantvillage

# 4. 训练（Kaggle T4，约 90 分钟）
python -m kaggle kernels push -p kaggle/fbr --accelerator NvidiaTeslaT4

# 5. 评估（挂载训练输出，CPU 即可）
python -m kaggle kernels push -p kaggle/eval_fbr
```
