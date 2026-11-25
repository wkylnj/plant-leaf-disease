# PlantDoc 微调迁移到 Swin-T：脚本改动、冒烟结果与 Kaggle 配置

本文记录把已验证的 `ml/src/finetune_plantdoc.py`（ResNet18 五折交叉验证微调）
迁移到 Swin-T 的过程。**结论先行：脚本一行都不用改**，Swin-T 本来就已支持。

---

## 1. 脚本是否需要改：不需要

原以为脚本里可能硬编码了 `resnet18`，读完全文（670 行）确认它已经是模型无关的：

| 关注点 | 实现位置 | 结论 |
| --- | --- | --- |
| 模型名从哪来 | `finetune_plantdoc.py:419` | `args.model` → `ck["args"]["model"]` → 目录名兜底，三级回落 |
| 分类头如何切分学习率 | `finetune_plantdoc.py:90-100` `HEAD_PREFIX` | 已登记 `"swin_t": "head"`，与 `models.py` 里 `model.head = nn.Linear(...)` 一致 |
| 分类头维度 | `finetune_plantdoc.py:424-426` | 从 `ck["classes"]` 取 38 类，不写死 |
| 模型构建 | `finetune_plantdoc.py:266` | `build_model(model_name, num_classes, pretrained=False)` 后再 `load_state_dict` |
| 数据划分 | `make_stratified_folds` | 与模型无关 |
| 结果 json | `main()` 末尾 | 与模型无关 |

关键证据是 checkpoint 自带的元数据：

```
args.model = 'swin_t'        # 脚本会自动读成 swin_t
epoch = 8, val_acc = 0.9961
n_classes = 38               # 与 PlantVillage 分类头一致
img_size = 224
```

所以 `--run-dir runs/swin_t_224_15ep` 直接就能跑，`HEAD_PREFIX["swin_t"] = "head"`
也不是新加的——原脚本作者已经预埋了 Swin-T。

**本次没有修改 `finetune_plantdoc.py`，也没有动 `models.py` / `data.py` /
`eval_crossdomain.py` / `train_colab.py`。**

---

## 2. 冒烟测试（CPU）

命令：

```powershell
$env:PYTHONUTF8=1
cd "D:\Desktop\神经网络最终项目\ml"
python src/finetune_plantdoc.py --run-dir runs/swin_t_224_15ep `
    --folds 5 --folds-to-run 1 --epochs 1 `
    --limit-per-class 4 --limit-batches 20 `
    --batch-size 8 --num-workers 0 `
    --out runs/swin_t_224_15ep/plantdoc_ft_cv_smoke.json
```

实际输出（退出码 0）：

```
checkpoint : D:\Desktop\神经网络最终项目\ml\runs\swin_t_224_15ep\best.pt
模型       : swin_t（epoch 8，PV val_acc 0.9960919540229886）
设备       : cpu   torch 2.6.0+cpu
可用样本   : 108 张 / 27 个 PlantDoc 类（映射到 PV 的 27 类）
未映射剔除 : 2 张（['Tomato_two_spotted_spider_mites_leaf']）
配置       : epochs=1 bs=8 backbone_lr=0.0001 head_lr=0.001 ls=0.1 strong_aug=False

折划分（按类别分层）
折       训练    验证     训练类数     验证类数     来源(train/test)
0       87    21       27       21              87/0
1       83    25       27       25              83/0
2       86    22       27       22              86/0
3       91    17       27       17              91/0
4       85    23       27       23              85/0

[1/2] 零样本 baseline（权重原样加载，不微调）
  折 0  零样本 top-1=0.3810  作物级=0.4762  (n=21)

[2/2] 微调 1 折（每折重新从 PlantVillage 权重初始化）
  ── 折 0/0：训练 87，验证 21
    轮 1/1  loss=3.0831  val_top1=0.1905  val_crop=0.3810
    折 0 结果  top-1=0.1905  作物级=0.3810  top-3=0.4762  用时 20.2s
```

冒烟结论（只验证管线，**不构成任何准确率结论**）：

- ✅ Swin-T 权重加载成功，没有 shape 不匹配（38 类头对得上）
- ✅ 前向/反向/AdamW/余弦退火/AMP 分支在 CPU 上都走通
- ✅ 分类头切分生效：`lr_backbone=1e-6` 与 `lr_head=1e-6` 同步退火，说明两组参数
  都被 optimizer 接管（1 轮 cosine 退到 `eta_min = 1e-4 * 0.01`）
- ✅ 结果 json 正常落盘，`RESULT_JSON_BEGIN/END` 块可解析

微调后 top-1 反而从 0.3810 掉到 0.1905，**这是预期的**：只跑 1 轮、每类 4 张
（训练集 87 张）、20 个 batch，等于用极小样本把预训练特征打乱一轮。冒烟只看
"跑不跑得通"，不看数字。

耗时 20.2s / 20 batch（bs=8，160 张图当量）。同规模下 ResNet18 约 7s，比例
接近预期的 3 倍。

---

## 3. Kaggle kernel 配置

路径：`kaggle/swin_finetune/`（`run.py` + `kernel-metadata.json`）

```json
{
  "id": "wangkaiyu66/plant-leaf-swin-finetune",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["wangkaiyu66/plantdoc-pv-ckpt"],
  "machine_shape": "NvidiaTeslaT4"
}
```

结构与 `kaggle/plantdoc_ft/` 一致：下载 PlantDoc → 从 raw.githubusercontent 拉源码
（保持单一真源）→ 找 Swin-T 的 best.pt → 跑 5 折 + 同折零样本 baseline → 结果打到
stdout 的 `ALL_RESULTS_JSON_BEGIN/END`。

### 权重从哪来（重点）

`wangkaiyu66/plantdoc-pv-ckpt` 里**只有 resnet18 和 mobilenet_v3_large 的 best.pt，
没有 Swin-T**。`run.py` 的 `find_swin_ckpt()` 按以下顺序找：

1. **挂载的 Kaggle Dataset**（推荐）。遍历 `/kaggle/input/**/*.pt`，先按目录名里是否含
   `swin` 排序，再**读取每个候选的 `args.model` 确认等于 `swin_t`** 才采用。
   因为三份权重文件名都叫 `best.pt`，不看内容无法区分。
   做法：把 `ml/runs/swin_t_224_15ep/best.pt`（约 105MB）上传成**独立的**私有
   dataset（建议 `wangkaiyu66/plantdoc-swin-ckpt`），在本 kernel 的 Data 里挂载。
   **不要塞进 `plantdoc-pv-ckpt`**——那个数据集被 `plantdoc_ft` kernel 复用，
   改内容会让旧 kernel 的输入版本漂移。
2. **`SWIN_CKPT_URL` 环境变量**（GitHub Release 直链）。105MB 走 Release 附件可行，
   但不进仓库本体。下载后同样校验 `args.model` 与文件大小（>50MB）。
3. 都没有 → 打印明确的两条操作提示，`sys.exit(2)`，不静默跑错模型。

已用伪造 checkpoint 验证过这两条路径：三份 `best.pt` 混挂时能选中 swin 那份；
只有 resnet18 时正确返回 None 并输出提示。

超参与 ResNet18 那轮完全一致（`epochs=4, bs=64, backbone_lr=1e-4, head_lr=1e-3,
ls=0.1, 5 折`），这样两个 kernel 的数字可以直接并列。Swin-T 显存约 3 倍，T4 16GB 上
bs=64 @224 通常够用；若 OOM 把 `run.py` 里的 `BATCH_SIZE` 改成 32 即可（4 轮内不值得
再调 lr）。

**没有执行 `kaggle kernels push`**，按要求留给用户手动推送。

---

## 4. 为什么预期 Swin-T 微调后能超过 ResNet18 的 69.73%

先摆事实（同一套 5 折、同一套 38 类分类头、同一份 PlantDoc）：

| 指标 | ResNet18 | Swin-T | 差 |
| --- | --- | --- | --- |
| 零样本 top-1（@224） | 28.18% | **34.93%** | +6.75pp |
| 零样本 top-3 | 49.01% | 57.05% | +8.04pp |
| 零样本作物级 | 51.13% | 53.22% | +2.09pp |
| 微调后 top-1（5 折均值） | **69.73% ± 2.37%** | 待跑 | — |
| 微调后作物级 | 83.50% | 待跑 | — |

（Swin-T 另有 @352 = 37.09%、多分辨率平均 37.50% 的结果，见 `crossdomain.json` 与
相应 artifacts；本 kernel 统一走 @224，与 ResNet18 对齐。）

### 预估

- **top-1：71% ~ 74%**（中心值约 72.5%），即比 69.73% 高 1.5~4pp
- **作物级：84.5% ~ 86.5%**
- **5 折全部提升**（相对各自零样本）的概率很高；但**"Swin-T 每折都高于 ResNet18
  同折"不一定成立**，折间标准差 2.37pp 已经和预期增益同量级

### 理由

**支持超过的理由（主要）**

1. **起点高 6.75pp**。有监督微调是在预训练特征上做局部调整，不是从零学；
   起点差距在 4 轮、lr 1e-4 的小步微调下不会被抹平。ResNet18 从 28.18 提到 69.73
   是 +41.55pp，若 Swin-T 的"可提升空间"结构相似，34.93 + 相近增幅就落在 73% 附近。
2. **零样本 top-3 优势更大（+8.04pp）**。top-3 比 top-1 更能反映"特征空间里正确类
   排得够不够前"。Swin-T 的 top-3 领先幅度大于 top-1，说明它的错误更多是
   "排在 2、3 位"而非"完全跑偏"——这类错误正是微调最容易修正的。
3. **层次化窗口注意力对叶片病斑这类局部纹理更友好**。PlantDoc 的域偏移主要是
   拍摄条件（背景杂乱、光照、分辨率、病斑尺度不一），Swin 的窗口划分 + 移位窗口
   在多个尺度上建模，理论上比 ResNet18 的固定 7×7 感受野更能适应尺度变化。

**压制增益的理由（为什么不敢给更高）**

1. **微调会压缩模型间的差距，不是放大**。域适应文献里常见现象：强预训练模型
   零样本领先，微调后两者收敛到相近水平，因为目标域只有 2920 张、27 类，
   容量更大的模型更容易过拟合这 2336 张训练图（每折）。Swin-T 有 28M 参数，
   ResNet18 只有 11M，数据量不变时正则化压力更大。
2. **作物级只领先 2.09pp**，远小于 top-1 的 6.75pp。作物级是更粗的粒度，
   说明 Swin-T 的优势集中在"细分到具体病害"这一层；而微调阶段两类模型都会
   在细粒度上学到目标域分布，这部分优势可能被摊薄。
3. **只跑 4 轮、不做早停、不按验证折选轮**（脚本的刻意设计，避免验证集变成
   模型选择集）。轮数少对大模型更不利——Swin-T 通常需要更长 warmup 才能充分适配。
4. 本地 ResNet18 五折标准差 2.37pp，说明折间波动本身就有 2pp 量级。若真实增益
   只有 1~2pp，5 折配对比较未必能给出统计上干净的结论。

**如果结果低于 69.73% 怎么办**：最可能的原因是 4 轮不够或 bs=64 下 Swin-T 的有效
学习率偏小。届时优先试 `--epochs 6`（不改其他超参），而不是调 lr——保持与
ResNet18 那轮的可比性更重要。

---

## 5. 产物清单

| 路径 | 说明 |
| --- | --- |
| `kaggle/swin_finetune/run.py` | Swin-T 版 Kaggle kernel（含权重查找与缺失提示） |
| `kaggle/swin_finetune/kernel-metadata.json` | kernel 元数据，`machine_shape: NvidiaTeslaT4` |
| `ml/runs/swin_t_224_15ep/plantdoc_ft_cv_smoke.json` | 冒烟结果（脚本自动识别为 smoke 命名） |
| `ml/runs/swin_t_224_15ep/smoke_log.txt` | 冒烟完整日志 |
| `ml/artifacts/plantdoc_finetune_swin.md` | 本文 |

未改动的共享文件：`ml/src/finetune_plantdoc.py`、`ml/src/models.py`、
`ml/src/data.py`、`ml/src/eval_crossdomain.py`、`ml/src/train_colab.py`。
未执行任何 `git add` / `git commit`，也未执行 `kaggle kernels push`。
