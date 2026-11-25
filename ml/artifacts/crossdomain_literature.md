# 跨域泛化文献与开源方案调研

> 目的：本项目零样本跨域（PlantVillage → PlantDoc）停在 37.50%，想知道学术界和开源社区有没有能直接用的解法，重点看**不依赖目标域标签**的方法。
>
> 调研时间：2026-11-25。数据来源：arXiv / PMC / IEEE Xplore / OpenReview / GitHub。

---

## 1. 一句话结论

**有解，而且解法方向和我们六条证伪路线的教训完全吻合：必须让模型在训练时接触到「目标域的成像风格」，而不是在推理时修补输入。**

文献里最有效的三条路线——子域对齐（MSUN，PlantDoc 56.06%）、背景重组 + 对抗对齐（FBR+CDAN+E，+44pp）、少样本度量学习（TMPS）——共同点都是**动骨干 + 喂目标域信息**，和我们的 E-03（换头无效）、E-12（喂同源局部 −5.79pp）、E-14（喂目标域 +41.55pp）三组实验结论一致。

其中**背景重组（FBR）**被我们此前误判了：E-16「叶片裁剪」失败不代表背景替换无效，两者方向相反（详见 §5）。

---

## 2. 我们的现状与文献基线的可比性

| 设定 | 方法 | PlantDoc 准确率 | 备注 |
|---|---|---|---|
| 零样本（只训 PlantVillage） | 我们的 ResNet18 | 28.18% | 2920 张全量 |
| 零样本 | 我们的 Swin-T @352 | 37.09% | 单模型最佳 |
| 零样本 | 我们的 Swin-T 多分辨率平均 | **37.50%** | 当前最佳，零成本档 |
| 零样本 | MSUN 论文 Baseline (w/o DA) | 30.78% | 与我们的 28.18% 同量级，可直接对照 |
| 零样本 | MSUN 论文 D-Coral | 33.37% | +2.59pp vs 其 baseline |
| 零样本 | MSUN 论文 DANN | 39.71% | +8.93pp |
| 零样本 | MSUN 论文 DSAN | 42.87% | +12.09pp |
| 零样本 | **MSUN 论文 MSUN** | **56.06%** | +25.28pp，最高 |
| 有监督微调 | 我们的 ResNet18 五折 | 69.73% | 用了 PlantDoc 标签 |
| 少样本（10 张/类） | TMPS 论文 | macro-F1 +7.3 | 相对「源+目标混训」 |

**可比性说明**：MSUN 的 PVD→C-PD 实验用 20% PlantVillage 作源域、C-PD（裁剪版 PlantDoc）5601 张作目标域；我们用全量 PlantVillage + 未裁剪 PlantDoc 2920 张。数据集版本和源域规模不同，**数字不能直接比大小**，但「baseline 30.78% vs 我们 28.18%」说明难度同一档，其方法增益量级可参考。

---

## 3. 文献方案总览

### 3.1 MSUN：多表示 + 子域对齐 + 不确定性正则（最值得抄的方法）

- 出处：Wu X, Fan X, Luo P, et al. *From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild.* Plant Phenomics, 2023;5:0038. DOI 10.34133/plantphenomics.0038（[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10059679/)）
- 三个模块：
  1. **多表示模块（Inception Adaptation）**：用 1×1 / 3×3 / 5×5 三种卷积核并联，同时抓整体结构和细节。动机是实验室图只有一片叶子，田间图还有果子、树枝等无关信息，单表示不够。
  2. **子域对齐（LMMD，局部最大均值差异）**：不是对齐两个域的全局分布，而是**按类别对齐子域**。这是它比 DANN/CORAL 高的核心原因——全局对齐会让不同病害的簇混在一起，而 PlantVillage 的痛点是「同作物不同病」占错误 22~28%（我们 E-09 实测）。
  3. **不确定性正则（熵最小化，半监督）**：把决策边界推到低密度区域，抑制伪标签错误累积。
- 关键点：**非对抗训练**，前向 + 反向传播即可，比 DANN 好调、训练稳定。目标域数据**不需要标签**，只用无标注 PlantDoc 图像。
- 消融（PlantDoc）：多表示模块 +26.34%（对 Plant-Pathology 是 +26.34%，PlantDoc 见原文表）；子域对齐 +10.39pp；三个模块全开 56.06%。
- **没有官方开源代码**（GitHub 搜不到），需要自己实现。LMMD 实现不复杂（DSAN 的 LMMD 有公开实现可参考）。

### 3.2 FBR：背景重组（杠杆最大，且我们已有全部前置条件）

- 出处：Jeon W, Kim T, Choi S, et al. *Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by background recomposition.* Ecological Informatics, 2026（[OpenReview](https://openreview.net/forum?id=wmtdrE09fq)、[中文解读](https://m.ebiotrade.com/newsf/2025-12/20251226083548017.htm)）
- 做法两步：
  1. **FBR（Field Background Recomposition）**：用 **SAM** 分割叶片 → 形态学膨胀/腐蚀两次填洞 → 把叶片贴到从真实田间采集的背景图上（背景库 200+ 种，随机旋转 ±45° + 裁剪）。
  2. **再叠无监督域适应**：DDC / DCORAL / DANN / CDAN+E / DALN 任选。
- 关键数字（苹果叶，真实环境）：baseline **47.1%** → 仅 FBR **57.0%**（+9.9pp）→ DDC+FBR 57.7% → **CDAN+E+FBR 91.1%**（迁移上限 97.7%）。葡萄叶 57.4% → 81.0%。
- **为什么这条对我们最重要**：
  - 单独 FBR 就有 +9.9pp，是所有「纯数据侧」方法里杠杆最大的。
  - **我们不需要 SAM**：E-16 写的 `ml/src/eval_leafcrop.py` 里 `leaf_mask()` 已经用超绿指数 + 饱和度 + Otsu + 形态学 + 最大连通域实现了抠图，**实测 fallback 0/252，抠图 100% 成功**。直接复用即可。
  - 真实背景库：PlantDoc 的 2920 张图里裁掉叶片区域剩下的就是田间背景，或者用 `scripts/fetch_plantdoc.py` 已有的 HttpRange 方式只抽部分图。
  - 纯 CPU 也能做（合成图 + 存盘，不需要 GPU）。

### 3.3 TMPS：少样本目标感知度量学习

- 出处：Nogami T, Kagiwada S, Iyatomi H. *Robust Plant Disease Diagnosis with Few Target-Domain Samples.* arXiv:2510.12909，VCIP 2025（[arXiv](https://arxiv.org/abs/2510.12909)）
- 设定：**允许每类 10 张带标签的目标域样本**。用度量学习 + 优先采样（prioritized sampling）。
- 结果：比「源域 + 目标域混在一起直接训」高 **7.3 个 macro-F1 点**，比「源域预训练后微调」高 **3.6 点**，比 baseline 高 18.7 点。
- 对我们的意义：如果真能拿到用户实拍的每类 10 张照片，这条比全量微调更省标注。但**它是半监督，不是零样本**，报告里必须归到「有监督/少样本」那一档。

### 3.4 CORAL + 多任务学习（实现成本最低）

- 出处：Rustamova M, Kadyrov S, Baimukashev R. *Multi-Task Learning with CORAL-Based Feature Alignment for Cross-Domain Plant Disease Detection.* ICECCO 2026（[IEEE](https://xplorestaging.ieee.org/document/11488762/)）
- 做法：**同时预测作物种类 + 病害类别**（多任务），再用 **CORAL**（二阶统计量协方差对齐）对齐源域和目标域特征。目标域数据**不需要标签**。
- 结论：多任务一致优于单任务；CORAL 对齐进一步提升目标域准确率和 macro-F1；最强的变体是「自适应多尺度多任务」。
- 为什么这条和我们的数据特别契合：我们 E-09 实测**作物级准确率（51~58%）远高于病害级（28~35%）**，说明「作物」是模型已经掌握的强信号。把作物识别作为辅助任务显式建模，等于把这条已经学会的能力变成正则项，方向自然。
- 实现成本：CORAL loss 十几行（`||C_s - C_t||_F^2 / (4 d^2)`），多任务只是把分类头换成两个。**不需要改数据管线，不需要额外数据准备**。

### 3.5 其他相关工作（供报告引用）

| 方法 | 出处 | 关键数字 | 是否需要目标域标签 |
|---|---|---|---|
| RMNA（ROI-Mixup + 邻域对齐） | Li B, Iyatomi H. IEEE SMC 2025（[IEEE](https://xplorestaging.ieee.org/document/11343547/)） | 67.14% → 72.13%（+4.99pp），F1 73→80 | 否（transductive，训练时可用全部无标注测试图） |
| DANN + CORAL 混合 | Prasad P S, Agniraj S. IJERR 2024（[PDF](https://pdfs.semanticscholar.org/3790/97edf125368a73aaaccf5813c1d724137188.pdf)） | 91.39% acc | 否，但目标是「New Plant Village」不是 PlantDoc，难度不同 |
| MTL + CORAL | ICECCO 2026（同上 §3.4） | 一致提升 | 否 |
| 领域不变 + 边缘高效框架 | Hossain S, et al. IEEE Access 2026 | 跨域 69.98% | 否（对比学习 + 轻量对抗对齐 + 压缩） |
| 先验匹配双分支网络 | Gao X. 2026（[Semantic Scholar](https://www.semanticscholar.org/paper/b38a36a83a99ed90f4bcb670d6633ecd98908aa3)） | 优于 source-only / 差异对齐 / 对抗 / 图像翻译四类基线 | 否 |

### 3.6 开源仓库

| 仓库 | 内容 | 对我们有用吗 |
|---|---|---|
| [TaherPanbiharwala/mobilenetv2-plantdoc-adaptation](https://github.com/TaherPanbiharwala/mobilenetv2-plantdoc-adaptation) | MobileNetV2 两阶段微调，PlantDoc 真实图 48.3% | 参考，但是有监督微调路线 |
| [sedanurkilic/Cross-Domain-Plant-Disease-Classification](https://github.com/sedanurkilic/Cross-Domain-Plant-Disease-Classification) | EfficientNet-B0 + KATv2（原型交叉注意力），番茄 8 类，146 张微调 / 598 张测试 | **最有价值**，见 §4 |
| [supus4562/ood-generalization-adaptation](https://github.com/supus4562/ood-generalization-adaptation) | DANN/CDAN/DAN/SAM 的 PyTorch 实现合集 | 可直接抄 DANN/CDAN 的 loss 实现 |
| [hafeez381/ood-robustness](https://github.com/hafeez381/ood-robustness) | DAN/DANN/CDAN + ERM/DAN-DG/SAM | 同上，代码质量更规整 |

**没有找到** MSUN、FBR、TMPS、RMNA 的官方代码。

---

## 4. 一个独立复现者的结论，和我们的高度一致

[sedanurkilic 的仓库](https://github.com/sedanurkilic/Cross-Domain-Plant-Disease-Classification) 是独立研究者做 PlantVillage → PlantDoc 番茄 8 类跨域，结论几乎逐条对上我们的实验：

| 他的结论 | 我们的对应实验 |
|---|---|
| 「架构不是瓶颈」——KATv2（原型交叉注意力，balanced acc 0.3545）与 EfficientNet（0.3553）几乎相同 | **E-03 线性探针**：换掉分类头后同域 +0.14~0.25pp、跨域 ±0.65pp，分类头贡献接近零 |
| 「数据泄漏会把结果抬高约 0.10」——用测试集挑 epoch 得到 0.4510，CV-clean 只有 0.3595 | 我们 E-14 五折交叉验证、不做早停、不按验证折选轮次，正是为了避开这一点 |
| 「AdaBN 在骨干部分可训练时有害」——ep32 −0.015、ep60 −0.035 | **E-13 TTA**：BN 重估 −6.75pp、TENT −24pp |
| 「~0.35 平衡准确率天花板，focal loss / 多样性损失 / 架构改动都推不过去」 | **E-10 da_aug**、**E-11 集成**、**E-17 颜色对齐** 全部证伪 |
| 「瓶颈是 146 张微调池，不是损失函数或模型容量」 | **E-14**：换成 2920 张有标签 PlantDoc 后 28.18% → 69.73%（+41.55pp） |
| 失败模式：视觉相似混淆（bacterial_spot → septoria）、病程重叠（early_blight 晚期 → yellow_virus） | **E-09**：同作物不同病占错误 22~28%，高置信度错误 311 例 |

**这份独立复现给了我们报告一个很强的支点**：我们的六条证伪不是能力不足，而是这个任务在无标签条件下的客观边界——别人用完全不同的架构（EfficientNet + 自定义 Transformer）撞到同一个天花板（~0.35），我们撞到 0.375（多分辨率平均）。

---

## 5. 关键纠正：E-16「叶片裁剪失败」不能推出「背景替换无效」

这是本次调研最重要的发现，因为此前我们把这两件事混为一谈了。

| | E-16 叶片定位裁剪（我们做的，−3.57pp） | FBR 背景重组（文献做的，+9.9pp） |
|---|---|---|
| 输入变成什么 | 把叶片**从背景里裁出来**，背景被丢弃 | 把叶片**贴到目标域背景上**，背景换成田间的 |
| 训练分布 vs 目标域 | 训练是「无背景纯叶片」，目标是「有复杂背景」→ **差距变大** | 训练是「田间背景 + 叶片」，目标是「田间背景 + 叶片」→ **差距变小** |
| 改变了什么 | 只是缩小了视野范围 | 改变了成像风格（背景、光照、构图） |
| 结果 | 0.3373 → 0.3016 | 47.1% → 57.0% |

**两者方向完全相反。** E-16 的失败恰好反证了「背景是域偏移的主要载体」——因为把背景拿掉反而更差，说明模型依赖的正是「背景 + 叶片的整体成像风格」。那么反过来，**把源域叶片的背景换成目标域背景，就是在正面攻击这个载体**。

文献的 Grad-CAM 证据也支持：FBR 之后模型对病斑区域的关注度提升 2.3 倍，背景干扰减少 67%。

---

## 6. 建议的下一步（按性价比排序）

### 第一优先：CORAL 对齐 + 多任务（作物 + 病害）

- 成本：约 100 行改动（`ml/src/train_colab.py` 加 CORAL loss + 双分类头），**不需要新数据、不需要 GPU 之外的额外条件**，Kaggle 一次 15 轮约 90 分钟。
- 预期：文献里 D-Coral 单独 +2.6pp、DSAN 类子域对齐 +12pp；多任务再加成。保守估计零样本从 37.50% 提到 **42~48%**。
- 为什么先做它：唯一一条「不改数据管线、不依赖新数据、有明确文献支撑」的路线，失败成本最低。

### 第二优先：FBR 背景重组（杠杆最大）

- 成本：中等。抠图复用 `eval_leafcrop.py` 的 `leaf_mask()`（已验证 100% 成功）；背景库从 PlantDoc 取；合成 39170 张图存盘约 10~15 分钟 CPU。
- 预期：文献单独 FBR +9.9pp。我们的起点更高（37.50% vs 47.1% 的苹果），增益可能小一些，但方向明确。
- 可与第一优先叠加（FBR 之后再叠 CORAL，就是文献的完整配方）。

### 第三优先：LMMD 子域对齐（MSUN 核心）

- 成本：中高（要按类别对齐，需要伪标签 + 类别权重）。
- 预期：文献最高（56.06%），但实现复杂度也最高。建议在 CORAL 跑通、理解了对齐的作用之后再上。

### 不建议

- **DANN / CDAN 对抗训练**：调参成本高（对抗训练对学习率、判别器容量敏感），我们只有 Kaggle 免费额度，试错预算不够。文献里 DANN 单独 39.71%，不比 DSAN 的非对抗方案好。
- **RMNA / TMPS**：前者需要 transductive 设定（训练时能看全部测试图），后者需要每类 10 张标注，都不符合我们「零样本 + 无目标域标注」的定位。

---

## 7. 已知局限

1. 文献数字的数据集版本与我们不同（C-PD 裁剪版 5601 张 vs 我们未裁剪 2920 张；源域 20% PVD vs 全量 PVD），**不能直接比大小**，只能比「相对 baseline 的增益量级」。
2. MSUN / FBR / TMPS / RMNA 均**无官方代码**，复现存在实现差异风险。
3. 上述预期增益是基于文献外推，**我们自己的六条证伪说明这个任务的直觉外推经常失效**，必须实测。
4. 未检索中文期刊与学位论文，可能漏掉国内团队的工作。
5. 未评估这些方法的**推理成本**（MSUN 的多表示模块、FBR 的合成图训练都会增加训练时间），对本项目「CPU 可跑」的约束是额外负担。

---

## 8. 参考链接汇总

- MSUN：https://pmc.ncbi.nlm.nih.gov/articles/PMC10059679/ ｜ DOI 10.34133/plantphenomics.0038
- FBR + UDA：https://openreview.net/forum?id=wmtdrE09fq ｜ 中文解读 https://m.ebiotrade.com/newsf/2025-12/20251226083548017.htm
- TMPS：https://arxiv.org/abs/2510.12909
- CORAL + MTL：https://xplorestaging.ieee.org/document/11488762/
- RMNA：https://xplorestaging.ieee.org/document/11343547/
- DANN+CORAL 混合：https://pdfs.semanticscholar.org/3790/97edf125368a73aaaccf5813c1d724137188.pdf
- 独立复现仓库：https://github.com/sedanurkilic/Cross-Domain-Plant-Disease-Classification
- 域适应实现合集：https://github.com/supus4562/ood-generalization-adaptation ｜ https://github.com/hafeez381/ood-robustness
- PlantDoc 原始论文：https://arxiv.org/abs/1911.10317
