# 田间植物病害数据集调研

> 目的：PlantDoc 只有 2920 张、27 类，且「健康」类几乎全错（`06_Cherry_including_sour_healthy`
> top-1 = 0.00%）。需要找更大、更真实、覆盖更全的田间数据集，用来验证跨域结论是否成立、
> 以及能否作为第二/第三个跨域测试集。
>
> 调研时间：2026-11-25。结论按「能否立刻用」排序。

---

## 1. 一句话结论

**PlantWild v1（18542 张 / 89 类，含 30 个健康类）是唯一能补上 PlantDoc 最大短板的公开数据集**——
它能映射到我们 38 类中的 **32 类、共 7455 张**，而 PlantDoc 只有 27 类、2920 张，且健康类全军覆没。
PlantWild 的许可（CC BY-NC-ND 4.0）允许非商业研究使用，下载渠道已验证可通（HF Range 读 zip 中央目录成功）。

---

## 2. 数据集横向对比

数据来自 PlantSeg 论文 Table 1（Wei et al., 2024）+ 各自官方页面核实。

| 数据集 | 年份 | 图片数 | 类别数 | 作物数 | 田间 | 框 | 分割掩膜 | 许可 | 我们的可用性 |
|---|---|---|---|---|---|---|---|---|---|
| PlantVillage | 2015 | 54,309 | 38 | 14 | ✗ | ✗ | ✗ | CC BY-SA 3.0 | **已在用（源域）** |
| PlantDoc | 2020 | 2,598 | 27 | 13 | ✓ | ✗ | ✗ | CC BY 4.0 | **已在用（跨域测试）** |
| **PlantWild v1** | 2024 | **18,542** | **89** | 33 | ✓ | ✗ | ✗ | CC BY-NC-ND 4.0 | **推荐新增** |
| PlantWild v2 | 2026 | 11,488 | 115 | 34 | ✓ | ✗ | ✗ | CC BY-NC-ND 4.0 | 无健康类，作补充 |
| FieldPlant | 2023 | 5,170 | 27 | 3 | ✓ | ✗ | ✗ | CC BY 4.0 | 备选（作物只有 3 种） |
| PlantSeg | 2024 | 11,458 | 115 | 34 | ✓ | ✗ | **✓** | CC BY-NC-ND 4.0 | 分割任务，分类可用 |
| AgriNet | 2022 | ~160,000 | 423 | — | ✓ | — | — | 未明确 | 13.6 GB rar，待评估 |
| Tomato Disease | 2020 | 4,178 | 4 | 1 | ✗ | ✓ | ✗ | — | 单作物，不用 |
| GLDD | 2020 | 4,449 | 4 | 1 | ✗ | ✓ | ✗ | — | 单作物，不用 |

---

## 3. PlantWild（首选）

### 3.1 出处与授权

- 论文：Wei Tianqi, Chen Zhi, Huang Zi, Yu Xin. *Benchmarking In-the-Wild Multimodal Plant Disease
  Recognition and a Versatile Baseline.* **ACM MM 2024**. [arXiv:2408.03120](https://arxiv.org/abs/2408.03120)
- 项目页：<https://tqwei05.github.io/PlantWild>
- 代码：<https://github.com/tqwei05/MVPDR>
- 许可：**CC BY-NC-ND 4.0**（非商业、禁演绎 —— 课程作业属非商业研究，可用；不能再分发修改版）

### 3.2 下载渠道（三个，全部已验证可达）

| 渠道 | 地址 | 备注 |
|---|---|---|
| **HuggingFace** | `uqtwei2/PlantWild` | **已实测可读**，`plantwild.zip` 2708.5 MB、`plantwild_v2.zip` 1640.3 MB |
| Google Drive | `1s7FOoztTHvO03yVfw75pQY_kzZqvAckD`（v1）/ `1wJEMRaNNuYGDqq2IQraQVtkCRHTOIDLD`（v2） | 需要浏览器，脚本难自动化 |
| UQRDM | `https://cloud.rdm.uq.edu.au/index.php/s/5iTzoyby9Xobmq2`（密码 `plantwildv1`） | 学校网盘，速度未知 |

**HF 直链**（脚本可直接用）：
```
https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip
https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild_v2.zip
```

### 3.3 实测结构（用 HTTP Range 只读 zip 中央目录，未下载整包）

`plantwild.zip`（v1）：
- 18,637 个 zip 条目，18,542 张图，**89 类**
- 目录结构 `plantwild/images/<类别名>/<编号>.jpg`，另有 `plantwild/classes.txt`
- 每类张数 min=44 / median=212 / max=589，**不均衡比 13.4×**（比 PlantVillage 的 36.1× 温和）
- **含 30 个健康类**，命名规则是 `<作物> leaf`（如 `apple leaf`、`corn leaf`、`tomato leaf`）

`plantwild_v2.zip`（v2）：
- 11,604 个条目，11,488 张图，**115 类**
- 目录结构 `plantwild_v2/<类别名>/<作物>_<病名>_<来源>_<编号>.jpg`
- 每类张数 min=7 / median=81 / max=390，**不均衡比 55.7×**
- **没有健康类**（农业专家精修时把「干净叶片」剔除了，只留病害）

### 3.4 与我们 38 类的映射

映射表写在 `ml/src/plantwild_map.py`（`PLANTWILD_TO_PV` 字典）。

**能映射 32/38 类，对应 PlantWild v1 的 7455 张图**：

| PlantVillage 类别 | PlantWild v1 目录 | 张数 |
|---|---|---|
| 00_Apple_Apple_scab | apple scab | 292 |
| 01_Apple_Black_rot | apple black rot | 173 |
| 02_Apple_Cedar_apple_rust | apple rust | 308 |
| 03_Apple_healthy | apple leaf | 444 |
| 04_Blueberry_healthy | blueberry leaf | 281 |
| 05_Cherry_including_sour_Powdery_mildew | cherry powdery mildew | 114 |
| 06_Cherry_including_sour_healthy | cherry leaf | 286 |
| 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | corn gray leaf spot | 216 |
| 08_Corn_maize_Common_rust | corn rust | 237 |
| 09_Corn_maize_Northern_Leaf_Blight | corn northern leaf blight | 224 |
| 10_Corn_maize_healthy | corn leaf | 156 |
| 11_Grape_Black_rot | grape black rot | 229 |
| 14_Grape_healthy | grape leaf | 201 |
| 15_Orange_Haunglongbing_Citrus_greening | citrus greening disease | 237 |
| 17_Peach_healthy | peach leaf | 157 |
| 19_Pepper_bell_healthy | bell pepper leaf | 240 |
| 20_Potato_Early_blight | potato early blight | 227 |
| 21_Potato_Late_blight | potato late blight | 240 |
| 22_Potato_healthy | potato leaf | 249 |
| 23_Raspberry_healthy | raspberry leaf | 180 |
| 24_Soybean_healthy | soybean leaf | 242 |
| 25_Squash_Powdery_mildew | squash powdery mildew | 281 |
| 26_Strawberry_Leaf_scorch | strawberry leaf scorch | 76 |
| 27_Strawberry_healthy | strawberry leaf | 199 |
| 28_Tomato_Bacterial_spot | tomato bacterial leaf spot | 280 |
| 29_Tomato_Early_blight | tomato early blight | 346 |
| 30_Tomato_Late_blight | tomato late blight | 295 |
| 31_Tomato_Leaf_Mold | tomato leaf mold | 239 |
| 32_Tomato_Septoria_leaf_spot | tomato septoria leaf spot | 220 |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | tomato yellow leaf curl virus | 171 |
| 36_Tomato_Tomato_mosaic_virus | tomato mosaic virus | 189 |
| 37_Tomato_healthy | tomato leaf | 226 |

**不可映射的 6 类**（PlantWild 没有语义一致的类别）：

| PlantVillage 类别 | PlantWild 候选 | 为什么不映射 |
|---|---|---|
| 12_Grape_Esca_Black_Measles | 无 | 黑麻疹是木质部病害，叶部症状与 leaf spot 混淆严重 |
| 13_Grape_Leaf_blight_Isariopsis_Leaf_Spot | grape leaf spot | 病名不完全一致，怕引入标签噪声 |
| 16_Peach_Bacterial_spot | peach leaf curl / peach scab | 病种不同 |
| 18_Pepper_bell_Bacterial_spot | bell pepper leaf spot | 病名不精确（可能是多种斑点病统称） |
| 33_Tomato_Spider_mites_Two_spotted_spider_mite | 无 | 虫害不是病害 |
| 34_Tomato_Target_Spot | tomato anthracnose | 病种不同 |

### 3.5 为什么这个数据集对我们特别有价值

1. **补上健康类**：PlantDoc 上 `06_Cherry_including_sour_healthy` top-1 = 0.00%、
   `37_Tomato_healthy` = 0.00%。PlantWild 有 13 个能对上的健康类、共约 2600 张。
   如果模型在 PlantWild 的健康类上表现明显好于 PlantDoc，就能证明「健康类全错」不是模型缺陷，
   而是 **PlantDoc 的健康类图像与 PlantVillage 的成像差异过大**（田间多叶、带土、有阴影）。
2. **样本量翻倍**：7455 vs 2920 张，每类 76~444 张（PlantDoc 是 2~500 张、多数 <100）。
3. **可做「第二个独立测试集」**：跨域结论如果只在 PlantDoc 上成立，会有人质疑是数据集特例。
   在 PlantWild 上复现同样的「零样本 30% 上下 + 提分辨率有效 + 微调大涨」，结论就稳得多。
4. **类别更多更难**：89 类 vs 27 类，且含大量近缘病（apple scab / apple rust / apple black rot），
   比 PlantDoc 更考验细粒度判别。

### 3.6 已知风险

- **CC BY-NC-ND 4.0 的 ND（NoDerivatives）条款**：不能再分发「修改过的」数据集。我们只做本地评估、不发布，
  不违反；但**不能把 PlantWild 的图打进课程报告的公开仓库**。
- **爬虫来源**（Google Images / Ecosia / Baidu Images）→ 标注质量不如人工采集，论文自己也说初始 5 万张清洗到 1.8 万张。
- **类别命名与我们不完全一致**：32 类映射是我按病名人工判定的，有主观成分。论文没给官方映射表。
- **v1 与 v2 不能混用**：v2 去掉了健康类，且类别名格式不同（带来源后缀）。

---

## 4. PlantWild v2（补充，非首选）

- 11,488 张 / 115 类，农业专家精修，类别比 v1 多 26 个。
- **没有健康类** → 不能用来验证「健康类全错」这个最有价值的发现。
- 能映射到我们 38 类中的约 26 类（无健康类）。
- 用途：如果 v1 的结论成立，可用 v2 做「类别更多时是否依然成立」的稳健性检查。

---

## 5. FieldPlant（备选）

- 论文：Moupojou E, Tagne A, Retraint F, et al. *FieldPlant: A Dataset of Field Plant Images for Plant
  Disease Detection and Classification With Deep Learning.* **IEEE Access 2023**.
  [DOI 10.1109/ACCESS.2023.3263042](https://doi.org/10.1109/ACCESS.2023.3263042)
- **5,170 张田间图 / 27 类 / 8,629 个标注叶片**，由植物病理学家监督标注（比 PlantDoc 标注质量高）。
- 下载：Kaggle `bloox2/fieldplant`（5.9 GB，Roboflow 导出格式）、`manhhoangvan/fieldplant`。
- **硬伤：只有 3 种作物**（论文里是玉米、木薯等），无法覆盖我们 14 种作物。
- 价值：作为「标注质量更好」的对照，验证 PlantDoc 的错误里有多少是标注噪声造成的。

---

## 6. PlantSeg（分割任务，分类可用）

- 论文：Wei Tianqi, et al. *PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation.*
  [arXiv:2409.04038](https://arxiv.org/abs/2409.04038)，**Scientific Data 13:205 (2026)**。
- 11,458 张 / 115 类 / 34 种作物，**每张带病斑分割掩膜**。
- 下载：Zenodo <https://zenodo.org/records/13958858>、Kaggle `weitianqi/plantseg`。
- **独特价值**：有分割掩膜 → 可以回答我们一直没法回答的问题「模型到底看的是不是病斑」。
  我们 E-7 用 Grad-CAM 只能得到弥散热力图，有了掩膜就能算 **IoU**（热力图 vs 真实病斑）。
  这是把 E-7 从「定性观察」升级成「定量指标」的唯一途径。
- 注意：分类任务用它要自己从掩膜裁出叶片/病斑。

---

## 7. AgriNet（规模最大，待评估）

- 论文：Al Sahili Z, Awad M. *The power of transfer learning in agricultural applications: AgriNet.*
  **Front Plant Sci. 2022;13:992700**. [PMC9794606](https://pmc.ncbi.nlm.nih.gov/articles/PMC9794606/)
- **约 160,000 张 / 423 类 / 19 个地理区域**，是目前最大的农业图像集合。
- 下载：HuggingFace `zahraa/AgriNet`，**`AgriNet.rar` 13.6 GB**。
- **问题**：①13.6 GB 单文件 rar，解压需要 ~20 GB 空间；②423 类里大量是「植物物种」而非「病害」，
  与我们 38 类病害的映射关系复杂；③论文没给明确的类别清单。
- 结论：**暂不下载**。如果 PlantWild 用完后还需要更多数据，再评估。

---

## 8. 建议的下一步（按性价比排序）

| 优先级 | 动作 | 成本 | 预期价值 |
|---|---|---|---|
| **1** | 下载 PlantWild v1（2.7 GB），按 `plantwild_map.py` 抽 32 类 7455 张，跑现有模型的零样本跨域 | 下载 30 min + 推理 15 min（CPU） | **验证「健康类全错」是数据问题还是模型问题**；给出第二个独立跨域数字 |
| 2 | 同一批 PlantWild 图跑分辨率扫描（224/288/352） | CPU 45 min | 验证「提分辨率有效」是否跨数据集成立 |
| 3 | 下载 PlantSeg（Zenodo），用掩膜算 Grad-CAM 的 IoU | 下载 + 实现约 3 h | **把 E-7 从定性升级为定量**，报告里最硬的一条 |
| 4 | FieldPlant 作为标注质量对照 | 5.9 GB + 推理 | 量化 PlantDoc 的标注噪声影响 |
| 5 | AgriNet | 13.6 GB | 暂缓 |

**注意**：PlantWild 是 CC BY-NC-ND，**不要提交到公开 GitHub 仓库**，只在本地评估。
抽取脚本产物应加进 `.gitignore`。

---

## 9. 参考链接

- PlantWild 项目页：<https://tqwei05.github.io/PlantWild>
- PlantWild 论文：<https://arxiv.org/abs/2408.03120>
- PlantWild HF：<https://huggingface.co/datasets/uqtwei2/PlantWild>
- MVPDR 代码：<https://github.com/tqwei05/MVPDR>
- PlantSeg 论文：<https://arxiv.org/abs/2409.04038>
- PlantSeg Zenodo：<https://zenodo.org/records/13958858>
- PlantSeg GitHub：<https://github.com/tqwei05/PlantSeg>
- FieldPlant 论文：<https://doi.org/10.1109/ACCESS.2023.3263042>
- FieldPlant Kaggle：<https://www.kaggle.com/datasets/bloox2/fieldplant>
- AgriNet 论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9794606/>
- AgriNet HF：<https://huggingface.co/datasets/zahraa/AgriNet>
- PlantDoc 论文：<https://arxiv.org/abs/1911.10317>
- 农业 CV 数据集汇总（2026）：<https://dev.to/harpreetsahota/top-agriculture-computer-vision-datasets-in-2026-crop-weed-segmentation-plant-phenotyping-3d-1icp>
