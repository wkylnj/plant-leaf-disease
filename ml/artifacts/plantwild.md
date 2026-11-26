# PlantWild 跨域评估：健康叶是独立的失效模式

## 一句话结论

换一个更难的田间数据集（PlantWild v1，7453 张 / 32 类）后，跨域准确率从 PlantDoc 的 28.18% 降到 **20.06%**（ResNet18）/ 34.93% 降到 **25.75%**（Swin-T），说明 PlantDoc 的 28~37% 并不是「田间照片的天花板」，而是**特定数据集上的成绩**。更重要的发现是：**健康叶的准确率稳定比病害叶低 6.6pp**，而且两个架构完全一致，这是一个与模型无关的数据/任务层面问题。

---

## 1. 为什么要做这个实验

PlantDoc 上有一个一直解释不通的现象：**所有「健康」类的 top-1 都是 0%**（`06_Cherry_including_sour_healthy` 57 张全错、`37_Tomato_healthy` 52 张全错）。当时有两种可能：

1. **模型缺陷**——模型学到的表示无法表达「健康」这个概念；
2. **PlantDoc 的健康叶图太脏**——田间多叶、带土、有阴影，模型从没在训练中见过「脏的健康叶」。

PlantDoc 里健康类只有零星几十张，无法区分这两者。PlantWild v1 含 **30 个健康类**（命名规则 `<作物> leaf`），其中 **12 个能对上我们的 38 类、共 2860 张**，样本量足够做统计判断。

> 数据集来源与映射表的完整推导见 `ml/artifacts/datasets_survey.md` 与 `ml/src/plantwild_map.py`。

---

## 2. 实验设置

| 项目 | 值 |
|---|---|
| 源域 | PlantVillage，39170 张 / 38 类（仅用训练集） |
| 目标域 | PlantWild v1 子集，**7453 张 / 32 类**（6 类无对应，见下） |
| 健康类 | 12 类，2860 张 |
| 病害类 | 20 类，4593 张 |
| 模型 | ResNet18 @224（epoch 11）、Swin-T @352（epoch 8） |
| 微调 | **无**，严格零样本 |
| 预处理 | 与训练验证集一致（Resize 短边 + CenterCrop + ImageNet 归一化） |
| 指标 | top-1 / top-3 / 作物级（同作物即算对）/ 平均置信度 |

**无对应的 6 类**（映射时排除，理由见 `plantwild_map.py` 的 `REJECTED`）：
`12_Grape_Esca_Black_Measles`、`13_Grape_Leaf_blight_Isariopsis_Leaf_Spot`、`16_Peach_Bacterial_spot`、`18_Pepper_bell_Bacterial_spot`、`33_Tomato_Spider_mites`、`34_Tomato_Target_Spot`。

**一个实现细节值得记下来**：标签索引必须按 `ck["classes"]` 的顺序建立，不能用 `sorted(目录名)`。后者会让模型输出通道与标签错位——`eval_crossdomain.py` 早期就踩过，top-1 从 33.73% 掉到 6.75%。

---

## 3. 主结果

### 3.1 两个数据集对比（同一批权重，零样本）

| 模型 | PlantDoc top-1 | PlantWild top-1 | 变化 | PlantDoc 作物级 | PlantWild 作物级 | 变化 |
|---|---|---|---|---|---|---|
| ResNet18 @224 | 0.2818 | **0.2006** | **−8.12pp** | 0.5113 | 0.3690 | **−14.23pp** |
| Swin-T @352 | 0.3493 | **0.2575** | **−9.18pp** | 0.5322 | 0.4357 | −9.65pp |

**PlantWild 明显比 PlantDoc 更难**，且作物级掉得比病害级更多（ResNet18 掉 14.23pp）。这意味着在更真实的田间图上，模型连「这是什么作物」都开始失守。

### 3.2 PlantWild 上的完整指标

| 模型 | top-1 | top-3 | 作物级 | 平均置信度 |
|---|---|---|---|---|
| ResNet18 @224 | 0.2006 | 0.3541 | 0.3690 | 0.4057 |
| Swin-T @352 | 0.2575 | 0.4320 | 0.4357 | 0.4089 |

Swin-T 在 PlantWild 上仍领先 ResNet18 **+5.69pp**（top-1），与 PlantDoc 上的 +6.75pp 量级一致——**Transformer 的优势跨数据集稳定存在**，不是 PlantDoc 的偶然。

---

## 4. 核心发现：健康叶是独立的失效模式

### 4.1 健康类 vs 病害类

| 模型 | 健康 n=2860 top-1 | 病害 n=4593 top-1 | 差距 | 健康 作物级 | 病害 作物级 |
|---|---|---|---|---|---|
| ResNet18 @224 | **0.1601** | 0.2258 | **−6.57pp** | 0.2552 | 0.4398 |
| Swin-T @352 | **0.2164** | 0.2830 | **−6.66pp** | 0.2993 | 0.5206 |

**两个架构的差距几乎完全相同（−6.57 vs −6.66pp）**，这不是某个模型的偏差。而且**作物级差距更大**（−18.5pp / −22.1pp）——说明健康叶连作物都认不出来。

### 4.2 健康叶被判到哪里去了

ResNet18 在 2860 张健康叶上的预测分布：

| 预测类 | 次数 | 占比 | 是否健康类 |
|---|---|---|---|
| `19_Pepper_bell_healthy` | 699 | **24.4%** | 是 |
| `26_Strawberry_Leaf_scorch` | 361 | 12.6% | 否 |
| `35_Tomato_TYLCV` | 259 | 9.1% | 否 |
| `30_Tomato_Late_blight` | 229 | 8.0% | 否 |
| `11_Grape_Black_rot` | 196 | 6.9% | 否 |
| `04_Blueberry_healthy` | 138 | 4.8% | 是 |
| `03_Apple_healthy` | 136 | 4.8% | 是 |
| `23_Raspberry_healthy` | 131 | 4.6% | 是 |

- **48.6% 判成某个健康类，51.4% 判成某个病害类**——健康叶有一半被误判成「有病」。
- **`19_Pepper_bell_healthy` 是一个「吸铁石」类**，单独吸走 24.4% 的所有健康叶。PlantVillage 里的青椒健康叶是「灰底 + 一片均匀绿叶」，正好是所有「普通绿叶」的视觉均值，于是成了默认答案。

### 4.3 逐健康类的去向（ResNet18）

| 健康类 | n | 判对 | top-1 误判去向 | 作物级 |
|---|---|---|---|---|
| `19_Pepper_bell_healthy` | 240 | **51.2%** | —（唯一判得动的） | 54.6% |
| `23_Raspberry_healthy` | 180 | 38.9% | — | 38.9% |
| `10_Corn_maize_healthy` | 155 | 38.7% | — | 40.0% |
| `14_Grape_healthy` | 201 | 24.9% | — | 44.3% |
| `04_Blueberry_healthy` | 281 | 20.6% | Grape_Black_rot 20.6%、Pepper_bell_healthy 16.7% | 20.6% |
| `03_Apple_healthy` | 444 | 13.1% | **Pepper_bell_healthy 22.3%**、Orange_HLB 13.3% | 14.2% |
| `27_Strawberry_healthy` | 199 | 12.1% | — | 62.3% |
| `24_Soybean_healthy` | 242 | **2.1%** | **Pepper_bell_healthy 39.3%**、Tomato_Late_blight 8.7% | **2.1%** |
| `22_Potato_healthy` | 249 | **2.0%** | **Pepper_bell_healthy 35.7%**、Tomato_Late_blight 14.9% | **6.4%** |
| `06_Cherry_including_sour_healthy` | 286 | **1.1%** | Pepper_bell_healthy 19.6%、Apple_healthy 17.8%、TYLCV 8.7% | **1.1%** |
| `37_Tomato_healthy` | 226 | **0.9%** | — | 46.9% |
| `17_Peach_healthy` | 157 | **0.0%** | — | **1.9%** |

**四个类连作物都认不出**（作物级 1.1%~6.4%）：Soybean、Peach、Cherry、Potato。它们不是「把病害认成健康」，而是**跨作物乱跳**——桃叶被当成别的作物，说明模型对这些图像完全没有可用特征。

### 4.4 为什么 PlantDoc 上健康类全错

现在可以回答了：**两个原因都存在，但主因是第二个。**

1. 模型确实不擅长表达「健康」——PlantVillage 的健康叶是 38 类中的 13 类，训练量本身就少（最少的 `22_Potato_healthy` 只有 110 张）；
2. **更关键的是「健康」这个概念在跨域时几乎失效**。健康叶没有任何病斑可看，模型只能依赖「整片叶子的颜色/纹理统计量」，而这恰恰是最容易被成像条件破坏的信息（见 `ml/artifacts/bg_causal.md`：只换背景、叶片像素不动，准确率就从 99.50% 掉到 83.50%）。

病害叶至少还有「病斑」这个相对成像无关的线索，健康叶什么都没有。**这是一个结构性困难，不是训练不充分。**

---

## 5. 与其它实验的关系

| 实验 | 结论 | 与本实验的关系 |
|---|---|---|
| E-2 手写特征（`handcrafted_ablation.md`） | 颜色主导同域，跨域仅 3.01% | 健康叶没有病斑，只能靠颜色 → 手工特征在健康类上应该更惨 |
| E-7 Grad-CAM（`gradcam.md`） | 注意力弥散居中，健康叶与病叶几乎相同 | 直接预示了健康类无法区分 |
| E-12 局部裁剪（`local_crop_training.md`） | 跨域 −5.79pp | 裁掉背景后健康叶更没线索 |
| bg_causal（`bg_causal.md`） | 只换背景掉 16pp | 解释健康叶为何在田间失效 |
| E-14 PlantDoc 微调（`plantdoc_finetune.md`） | 有监督微调 +41.55pp | 健康类是否也能被微调救回来，是本实验的后续 |

**四条独立实验指向同一机制**：模型依赖的是「整片叶子的全局颜色/纹理统计」，这个表示在健康叶上没有任何冗余（没有病斑兜底），所以在域偏移下最先崩。

---

## 6. 对项目的意义

1. **「28.18%」不能当作田间真实水平对外说**。换一个更难的公开田间数据集就掉到 20.06%。报告里应写成区间，并注明数据集。
2. **产品定位要避开健康类**。当前模型在田间健康叶上 16%（ResNet18）/ 22%（Swin-T），且一半会误报「有病」——对农户来说「没病说有病」比「有病说没病」更糟（会引发不必要的施药）。建议 UI 上对健康判定单独标注低置信度。
3. **`Pepper_bell_healthy` 吸铁石现象是一个可修的 bug**。24.4% 的健康叶被它吸走，说明该类在特征空间里占据了过大的体积。后续可以试：类别平衡采样、对该类做特征空间收缩、或在推理时对该类加先验惩罚。
4. **Swin-T 的优势跨数据集稳定（+5.69pp vs +6.75pp）**，这为「选 Swin-T 而不是 ResNet18」提供了第二个独立证据（第一个是 PlantDoc）。

---

## 7. 已知局限

1. **只有 32/38 类**，缺的 6 类里包括 `34_Tomato_Target_Spot`（281 张，PlantVillage 里的大类），会略微高估难度。
2. **PlantWild 是网络爬取数据**（Google/Ecosia/Baidu Images），标注噪声比 PlantDoc 更大，20.06% 里有部分是标注错误。
3. **两个数据集的类别分布不同**：PlantWild 健康类占 38.4%，PlantDoc 健康类占比极低。所以两个数字不能直接相减当「难度差」。
4. **只测了两个模型、单一分辨率**（ResNet18 @224、Swin-T @352），未做分辨率扫描。
5. **未做显著性检验**：7453 张样本量够大，但健康/病害分组内部的类别极不均衡（`19_Pepper_bell_healthy` 240 张 vs `26_Strawberry_Leaf_scorch` 76 张）。
6. **健康类判定的「正确」标准可商榷**：`03_Apple_healthy` 被预测成 `19_Pepper_bell_healthy`，从「有没有病」的角度其实是对的（两个都是健康叶），但按 38 类分类算错。若改用「健康/病害二分类」口径，健康类准确率会显著提高——这个口径值得单独做一版。

---

## 8. 产物

| 文件 | 内容 |
|---|---|
| `ml/src/plantwild_map.py` | 38 类 ↔ PlantWild 映射表（32 类可映射） |
| `ml/src/fetch_plantwild.py` | 从 HF zip 顺序抽取子集（7453 张，196s） |
| `ml/src/eval_plantwild.py` | 跨域评估，输出健康/病害分组 |
| `ml/src/analyze_healthy_confusion.py` | 健康叶误判去向分析 |
| `ml/data/raw/plantwild_subset/` | 7453 张，32 类目录（不入库，CC BY-NC-ND） |
| `ml/runs/resnet18_224_15ep/plantwild_224.json` / `.csv` | ResNet18 结果 |
| `ml/runs/swin_t_224_15ep/plantwild_352.json` / `.csv` | Swin-T 结果 |
| `ml/runs/resnet18_224_15ep/plantwild_healthy_confusion_224.json` | 健康叶去向明细 |
| `ml/artifacts/plantwild_v1_stats.json` | 数据集结构统计 |

---

## 9. 复现

```powershell
cd ml
# 1. 抽取子集（顺序读取 + 8MB 缓存，约 3.5 分钟）
python src/fetch_plantwild.py --out data/raw/plantwild_subset

# 2. 跨域评估
python src/eval_plantwild.py --run-dir runs/resnet18_224_15ep --img-size 224
python src/eval_plantwild.py --run-dir runs/swin_t_224_15ep --img-size 352

# 3. 健康叶误判去向
python src/analyze_healthy_confusion.py --run-dir runs/resnet18_224_15ep --img-size 224
```

> ⚠️ PlantWild 是 **CC BY-NC-ND 4.0**：课程作业属非商业研究可用，但**不能把图片提交到公开仓库**。`ml/data/raw/plantwild_subset/` 已在 `.gitignore` 中。
