# 背景替换训练增强（bg_replace）— 计划与进展

状态：**已暂停（未落地代码）**，2026-11-24。本文档记录已完成的调研、可行性验证与下一步实现方案。

## 动机（为什么只剩这一招）

- 同域 test **99.4%**，跨域 PlantDoc **28.2%**。
- Grad-CAM：四角注意力仅 **2.7%**，模型不是在偷背景，而是在用**整片叶子的全局颜色/纹理统计量**。
- 因此「扰动成像条件」类方法全部失败：ColorJitter/模糊/灰度 `da_aug` 跨域 **−2.22pp**；
  TTA（BN 重估/TENT/伪标签）10 组全负。
- 唯一有效：提高推理分辨率（**+2.20pp**）、多模型集成（**+0.93pp**）。
- 结论：必须**切断实验室成像条件这条捷径本身**。把 PlantVillage 叶片前景抠出来贴到真实田块
  背景上，模型无法再依赖「摆拍背景 + 均匀光照」的统计量，只能依赖叶片本身。

## 已完成的可行性验证（本机实测）

1. 依赖：`cv2 5.0.0` / `numpy 2.5.2` / `scipy 1.18.1` / `torch 2.6.0+cpu` 均可用（Kaggle 上
   scipy 随 sklearn 预装，故抠图用 **scipy.ndimage** 而不用 cv2，减少 Kaggle 侧依赖风险）。
2. `ml/data/processed/plantvillage/.prepared` 已存在，本地冒烟不会触发重新下载。
3. `ml/data/raw/plantdoc/` 有 2670 张真实照片（train/test 两棵树、`<类名>/*.jpg`），可直接作背景池。
4. **HF 侧只有整包 zip**：`LamTNguyen/PlantDoc` 的 siblings 只有 `.gitattributes` 与
   `PlantDoc.zip`（实测 **939 MB / 2924 entries / 2922 张图**），没有分卷或子目录。
5. **已实测走通「只抽少量图、不整包下载」的路径**：用自写的 `HttpRangeFile`（`urllib` + `Range`
   header + 1 MB 块缓存，约 40 行无第三方依赖）套 `zipfile.ZipFile`，读取中央目录 + 抽 2 张图
   **共 8.8 s、流量约数 MB**（本地直连 HF：HEAD 2.7 s、目录 5.6 s）。因此 kernel 不需要下 895 MB，
   抽 300 张背景图预计 <1 min。失败时再回落 `urllib` 流式整包下载 + 抽样。

## 待实现方案（下一步照此写，全部逻辑内联进 `ml/src/train_colab.py`）

- `DEFAULTS` 新增（默认关，历史行为逐字节不变）：`bg_replace=False`、`bg_dir=""`、`bg_prob=0.5`、
  `bg_leaf_min=0.4`、`bg_leaf_max=0.9`、`bg_feather=4`、`bg_thr_k=2.0`、`bg_min_area=0.05`、
  `bg_max_area=0.90`、`bg_canvas_scale=1.5`、`bg_max=0`；全部写入 `summary.json`。
- 抠图：四边 2 px 条带的中位色为背景估计 → 逐像素 RGB 距离 → 阈值
  `max(12, p95(边界距离) * bg_thr_k)`（相对阈值，自动适配明暗）→ 开闭运算 + `label` 取最大连通域
  + `binary_fill_holes` → 面积比不在 `[bg_min_area, bg_max_area]` 或主连通域占比过低 → **回退原图**。
- 羽化：`gaussian_filter(mask, sigma=bg_feather/2.5)` 得 alpha，3~5 px 软边。
- 合成：背景随机缩放（覆盖画布后随机取窗）到 `img_size*bg_canvas_scale` 画布；叶片按 alpha bbox 裁切、
  缩放至占画布 40%~90%、随机偏移（允许出画布后裁切）；alpha 混合贴回。
- 数据：`BGReplaceDataset` 包 `ImageFolder`，只在 **train** 侧包裹；`val`/`test` 的 `build_transforms`
  一行不动（用新旧版本 `repr(transform)` 逐项 diff 证明）。背景路径在 `__init__` 一次性 `rglob` 建索引。
- 多 worker 随机性：不用全局 seed，首个 `__getitem__` 时用 `torch.initial_seed() ^ (os.getpid()<<16)`
  初始化各自的 `random.Random` / `np.random.default_rng`。
- 训练前跑一次 `compose_probe(n=200)` 单进程统计抠图成功率/平均面积占比，写进 `summary.json`。

## 下一步最省事的顺序

1. 按上面实现 `train_colab.py`（约 200 行），本地：关开关 1 epoch 2 batch 对照 + 开开关冒烟。
2. `ml/src/viz_bg_replace.py` 出 8 张网格 → `ml/artifacts/bg_replace_samples.png`，**先看抠图质量再决定是否训练**。
3. 照 `kaggle/local_crop/` 建 `kaggle/bg_replace/`（`id: wangkaiyu66/plant-leaf-bg-replace`，
   `run_name=mobilenet_v3_large_bgreplace_224_15ep`），`kaggle kernels push -p kaggle/bg_replace --accelerator NvidiaTeslaT4`。

## 已知风险

- 抠图质量是命门：PlantVillage 有些图背景带纹理/阴影/叶片贴边，掩码易破；若成功率 <85% 或边缘明显，
  该实验结论不可信，应先人工看网格图。
- 合成图与真实 PlantDoc 仍有域差（叶片尺度/姿态/光照方向），可能只提升部分类。
- 训练分布与 val/test 分布不一致（val/test 仍是实验室图），**val 精度会掉**，判定只能用跨域 PlantDoc。
- 15 轮 CPU 不可行，必须上 Kaggle；免费 T4 排队很挤。
