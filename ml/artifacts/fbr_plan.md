# FBR 背景重组实验计划

> 状态：管线已打通并通过冒烟验证，等待 Kaggle 全量训练。
> 日期：2026-11-25

## 1. 方法与文献依据

**FBR（Field Background Recomposition）**：把 PlantVillage 的实验室摆拍叶片抠出来，
贴到 PlantDoc 的真实田间背景上，让模型在**训练时**就见到田间成像风格。

文献：Jeon et al., *Ecological Informatics* 2026。苹果叶跨域 47.1% → 57.0%（+9.9pp），
再叠 CDAN+E 达到 91.1%。核心洞察：**域偏移的主要载体是背景与成像风格，不是病斑本身**。

这个判断在本项目里有直接的反证：E-16「叶片裁剪」实验掉了 3.57pp
（见 `ml/artifacts/local_crop_plan.md`）。裁剪 = 训练时把背景丢掉、测试时背景又回来了，
域差反而更大；FBR = 训练时就把背景换成目标域的，域差变小。两者方向相反、结果相反，
互为印证。

当前基线（供对比）：

| 模型 | 同域 test acc | PlantDoc 零样本 |
|---|---|---|
| ResNet18 | 99.41% (`runs/resnet18_224_15ep`) | 28.18% |
| Swin-T | — | 37.50%（多分辨率平均） |

FBR 实验目标是 ResNet18 的 PlantDoc 零样本从 28.18% 往上走。

## 2. 管线

```
PlantDoc 原始图 ──make_bg_library.py──> data/raw/plantdoc_bg/  (bg_NNNN.jpg + manifest.json)
                                              │
PlantVillage/train ───────────────────────────┤
                                              ↓
                                  make_fbr_dataset.py（抠叶 → 贴田间背景）
                                              ↓
                        data/processed/pv_fbr/<类>/*.jpg   ← 扁平结构
                                              ↓
                                  prepare_fbr_marker.py（切 train/val + 写 .prepared）
                                              ↓
                        data/processed/pv_fbr/{train,val,test}/<类>/*.jpg
                                              ↓
                        train_colab.py --data-root data/processed/pv_fbr
```

### 2.1 是否修改了 train_colab.py

**没有改，一行都没动。** 理由：

`train_colab.py:129-135` 的 `download_and_prepare()` 判断只有一处：

```python
marker = root / ".prepared"
if marker.exists():
    log(f"数据已就绪: {root}")
    return root
```

只要 `data_root/.prepared` 存在，就**完全跳过** HuggingFace 下载。所以
「指向 pv_fbr 会不会误触发下载」的答案取决于标记文件在不在，不需要 `--skip-data`。

但 `main()`（`train_colab.py:435-449`）紧接着要求 `train/`、`val/`、`test/` 三个
ImageFolder 子目录，而 `make_fbr_dataset.py` 输出的是**扁平** `<类>/*.jpg`。
两者对不上——这才是真正需要补的东西。

新增 `ml/src/prepare_fbr_marker.py` 做三件事：

1. 把扁平合成图按 `--val-ratio`（默认 0.1，与 PlantVillage 的划分比例一致）
   切成 `train/` + `val/`，用**硬链接**（跨盘符失败自动回退复制，不额外占磁盘）；
2. 把原始 `plantvillage/test` 硬链接进 `test/`——因为合成图的源图只来自
   `plantvillage/train`，PlantVillage 的 test 是另一批原始图。**FBR 的 test
   绝不能放合成图**，否则 `train_colab.py` 结尾打的 test acc 就成了「合成图测试」；
3. 写 `.prepared` 标记（内容是 JSON，带 split 统计，便于事后核对）。

脚本可重复运行（幂等）：输出目录就是 `fbr_root` 本身，第二次跑时会多出
`train/` `val/` `test/` 三个目录，脚本已把它们排除在「类别」之外，不会误报
类别不一致。已存在的文件直接跳过（硬链接不会重复占空间）。

这样 FBR 与基线走**完全同一条代码路径**，唯一变量是 `--data-root`，可比性最强。

### 2.2 同域评估口径（重要）

`train_colab.py` 结尾的 test acc 用的是 `pv_fbr/test/`，也就是原始 PlantVillage test
（因为上面第 2 点），数字本身没问题；但 `best.pt` 是按**合成图 val** 选出来的，
与基线的「原始 val」选点口径不同。所以同域指标应统一用：

```powershell
cd ml
python src/eval_ckpt.py --run-dir runs/<run名> --data-root data/processed/plantvillage
```

跨域评估（不变）：

```powershell
python src/eval_crossdomain.py --run-dir runs/<run名> --plantdoc-root data/raw/plantdoc --split all
```

## 3. 冒烟验证

本机 torch 2.6.0+cpu、无 GPU，规模压到最小。用的是一份小样本合成数据
`data/processed/pv_fbr_smoke`（228 张，`per_image=2`）。

### 3.1 数据准备

```
[fbr-prep] FBR 类别 38 个，输出到 ...\ml\data\processed\pv_fbr_smoke
[fbr-prep] 类别与原始 PlantVillage 一致 ✓
[fbr-prep] train=190 val=38（val_ratio=0.1）
[fbr-prep] test=10878（来自原始 PlantVillage test）
[fbr-prep] 已写标记 ...\pv_fbr_smoke\.prepared
[fbr-prep] train/ 类别 ✓ (38 类, 190 张)
[fbr-prep] val/ 类别 ✓ (38 类, 38 张)
[fbr-prep] test/ 类别 ✓ (38 类, 10878 张)
```

### 3.2 不触发下载（关键验证）

命令：

```powershell
python src/train_colab.py --data-root data/processed/pv_fbr_smoke `
  --out-dir runs_smoke --run-name fbr_pipeline_smoke `
  --epochs 1 --limit-batches 3 --batch-size 8 --num-workers 0 --no-export-onnx
```

**本机根本没有安装 `datasets` 包**（`python -c "import datasets"` → ImportError），
而 `download_and_prepare()` 里唯一的下载路径就是 `from datasets import load_dataset`。
训练全程没有报 ImportError，说明下载分支确实被 `.prepared` 短路掉了。

### 3.3 训练与存 checkpoint

`runs_smoke/fbr_pipeline_smoke/` 产出：

| 文件 | 大小 | 说明 |
|---|---|---|
| `best.pt` | 44.9 MB | ResNet18 权重 + 38 类清单 |
| `config.json` | 561 B | `data_root: "data/processed/pv_fbr_smoke"` |
| `classes.json` | 1232 B | 38 类 |

`config.json` 里 `data_root` 指向 FBR 目录，证明训练读的是合成图。

### 3.4 训练样本确实是合成图

```
n_train 190
0 00_Apple_Apple_scab -> ...\00_Apple_Apple_scab\00075aa8-...___FREC_Scab 3335_fbr0.jpg
0 00_Apple_Apple_scab -> ...\00075aa8-...___FREC_Scab 3335_fbr1.jpg
...
all have _fbr: True
n_val 38 all _fbr: True
```

train/val 的**每一条**路径都带 `_fbr` 后缀（`make_fbr_dataset.py:212` 的命名规则），
即全部是 `make_fbr_dataset.py` 合成出来的，没有混入原始 PlantVillage 图。

## 4. Kaggle 配置

路径：`kaggle/fbr/`（`run.py` + `kernel-metadata.json`）

- id：`wangkaiyu66/plant-leaf-fbr`
- `enable_gpu: true`，`enable_internet: true`，`machine_shape: NvidiaTeslaT4`
- 超参与 `runs/resnet18_224_15ep` **完全一致**（resnet18 / 15ep / bs64 / lr3e-4 /
  img 224 / strong_aug），唯一变量是 `--data-root`

### 4.1 数据获取方式

| 数据 | 来源 | 说明 |
|---|---|---|
| PlantVillage | HuggingFace `GVJahnavi/PlantVillage_dataset` | 直接调 `train_colab.py --prepare-only`，复用其下载+转换逻辑，保证目录结构与本地一致 |
| PlantDoc | HuggingFace `LamTNguyen/PlantDoc`（900MB） | 官方源失败自动换 `hf-mirror.com`；解压后若多套一层目录会自动提上来 |

**为什么不用「上传 PlantDoc 成 Kaggle Dataset」**：`PlantDoc.zip` 有 939 MB，
上传+版本化很慢，且 HF 下载在 Kaggle 上通常几十秒到几分钟。用代码下载更省事，
也避免数据集版本和本地不一致。代价是每次跑 kernel 都要重下（见风险 R3）。

### 4.2 kernel 执行流程

1. `pip install datasets scipy`
2. 从 GitHub raw 拉 8 个脚本到 `/kaggle/working/src/`
   （`train_colab.py`、`make_bg_library.py`、`make_fbr_dataset.py`、
   `prepare_fbr_marker.py`、`eval_leafcrop.py`、`eval_crossdomain.py`、`data.py`、`models.py`）
3. 准备 PlantVillage → 下载解压 PlantDoc → 建背景库 → 合成 FBR → 切 split 写标记
4. 自检：`.prepared` 必须存在、抽样文件名必须含 `_fbr`（否则打警告）
5. `train_colab.py --data-root .../pv_fbr` 训练

**为什么拉 8 个脚本而不是 4 个**：`make_bg_library.py:33` 有
`from eval_leafcrop import leaf_mask`，而 `eval_leafcrop.py` 又 import
`data` / `models` / `eval_crossdomain`。少拉一个就 ImportError。

## 5. 风险点

### R1 背景库重建极慢（当前阻塞项）

观测：`make_bg_library.py --target 400` 从 11:22 启动，11 分钟后只保存了 **13 张**。

原因：拒收率极高。上一轮 manifest 的统计是

```json
{"source_images": 2922, "saved": 400,
 "rejects": {"leaf_too_dense": 1708, "white_studio": 354, "watermark": 184,
             "skin": 163, "too_small": 38, "too_dark": 4}}
```

即 2922 张里只有约 470 张通过筛选，要凑 400 张得几乎扫完全部源图。
单张要跑 `leaf_mask`（Otsu + 两轮形态学）+ 肤色统计，CPU 上约 0.5~1 秒/张，
所以一轮 15~25 分钟。**这不影响正确性，只是慢**；Kaggle 上 CPU 更强，会快一些。

### R2 背景库「是否真的像田间背景」未复核

`make_bg_library.py` 的设计已从「抠干净背景块」改成「整张田间照片当背景」
（脚本头注释 2026-11-25 的变更说明），理由是 PlantDoc 里大多是叶片特写，
抠不出干净背景。这个改动方向是对的，但**代价是背景里本来就有别的叶片**，
合成图可能出现「叶片叠叶片」的构图。

冒烟用的 `pv_fbr_smoke` 是旧背景库（310 张）合成的，已做量化与目视检查：

- 量化：合成图边缘 8px 环带 std = 43.7 / 67.2 / 37.8（纯色布应 < 5），
  说明背景确实是有纹理的真实照片，不是纯色布。
- 目视（`_fbr_verify_grid.png`）：背景是草地/土壤/其他叶片，叶片贴合成自然。

但发现**水印残留**：部分背景图里能看到 "Provided by ... photobucket" 之类的
图床水印。`make_bg_library.py` 的 `watermark` 过滤（拒掉 184 张）只检查
「底部近白条」，对压在图像中部、浅色的水印无效。影响：这部分合成图会带
与任务无关的文字纹理，轻微污染训练信号。全量合成后建议抽样人工过一遍。

### R2b 本机 CPU 竞争导致背景库重建极慢

2026-11-25 11:22 启动的 `make_bg_library.py --target 400` 到 11:38 只保存了
19 张。原因是同时有多个进程抢 CPU：

| PID | 进程 | 累计 CPU |
|---|---|---|
| 36644 | `make_bg_library.py --target 400` | 874s |
| 12724 | `eval_calibration.py --runs ...` | 2928s |
| 18328 | `train_lmmd.py --epochs 3 ...` | 300s |
| 7060 | 线性探针（冻结特征 + LogisticRegression） | 79s |

所以「慢」主要是资源竞争，不是脚本本身的问题。Kaggle 上 CPU 更宽裕。

### R3 Kaggle 每次重跑都要重下 ~1.7GB

PlantVillage 813MB + PlantDoc 939MB。kernel 的 `/kaggle/working` 不跨运行保留，
所以每次都要重下。若反复调参，建议后续把 `data/processed/pv_fbr` 打成
Kaggle Dataset 复用（首次训练产出后即可）。

### R4 合成图 val 选点与基线口径不同

见 §2.2。`best.pt` 按合成图 val 选，基线的 `best.pt` 按原始 val 选。
跨域对比时这个差异存在但通常次要（两者 val 都是同域数据）；
若要严格可比，可改为按固定 epoch 取权重。

### R5 `test/` 是硬链接，跨盘符会退回复制

`prepare_fbr_marker.py` 优先硬链接（省磁盘），失败自动 `shutil.copy2`。
10878 张 test 若走复制会多占约 250MB，不影响正确性。

## 6. 复现命令

本地（小规模验证）：

```powershell
cd ml
$env:PYTHONUTF8=1
# 1. 建背景库（约 15~25 分钟）
python src/make_bg_library.py --plantdoc-root data/raw/plantdoc --out data/raw/plantdoc_bg --target 400
# 2. 合成 FBR
python src/make_fbr_dataset.py --pv-root data/processed/plantvillage/train --bg-root data/raw/plantdoc_bg --out data/processed/pv_fbr --per-image 1 --size 224
# 3. 切 split + 写标记
python src/prepare_fbr_marker.py --fbr-root data/processed/pv_fbr
# 4. 冒烟
python src/train_colab.py --data-root data/processed/pv_fbr --out-dir runs_smoke --run-name fbr_smoke --epochs 1 --limit-batches 3 --batch-size 8 --num-workers 0 --no-export-onnx
```

Kaggle（**不要在本机执行 push**，由统一负责人操作）：

```powershell
python -m kaggle kernels push -p kaggle\fbr --accelerator NvidiaTeslaT4
```

训练完成后拉回：

```powershell
python -m kaggle kernels output wangkaiyu66/plant-leaf-fbr -p _kaggle_out\fbr
```

然后本地评估：

```powershell
cd ml
python src/eval_crossdomain.py --run-dir runs/resnet18_fbr_224_15ep --plantdoc-root data/raw/plantdoc --split all
python src/eval_ckpt.py --run-dir runs/resnet18_fbr_224_15ep --data-root data/processed/plantvillage
```

## 7. 交付清单

| 文件 | 状态 |
|---|---|
| `ml/src/prepare_fbr_marker.py` | 新增，已验证 |
| `kaggle/fbr/run.py` | 新增，未推送 |
| `kaggle/fbr/kernel-metadata.json` | 新增，未推送 |
| `ml/artifacts/fbr_plan.md` | 本文件 |

未改动：`train_colab.py`、`make_bg_library.py`、`make_fbr_dataset.py`、
`eval_crossdomain.py`、`data.py`、`models.py`。
