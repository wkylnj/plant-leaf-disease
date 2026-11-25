# LMMD 局部最大均值差异 — 实现方案与冒烟结果

状态：**代码与 Kaggle 配置已就绪，冒烟通过，全量训练未跑**（本机无 GPU）。
日期：2026-11-25。

要复现的论文：Wu X, Fan X, Luo P, Das Choudhury S, Tjahjadi T, Hu C.
*From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition
in the Wild.* Plant Phenomics 2023;5:0038.
全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC10059679/

## 1. 为什么只做 LMMD

MSUN 论文有三个模块：

| 模块 | 内容 | 本次是否实现 |
|---|---|---|
| ① 多表示模块 | Inception Adaptation，1×1/3×3/5×5 并联卷积 | ✗ |
| ② **LMMD** | 按类别对齐子域 | ✓ **只做这个** |
| ③ 不确定性正则 | 熵最小化 | ✗ |

一次只动一个变量。如果同时上三个模块，精度变了无法归因；而且模块①③本身都会
改变特征分布，会把 LMMD 的效果掩盖掉。

## 2. 方法：为什么必须按类别对齐

普通 MMD 对齐的是源域和目标域的**整体分布**。这在植物病害上会出问题：

- 病害类别之间**类间相似度高**（同一种作物的不同病害，病斑形态、叶色都相近）；
- 同一类别内部**类内变异大**（病程早晚、光照、拍摄距离差异巨大）。

全局对齐会强行把「番茄早疫病」和「番茄晚疫病」的簇拉到一起，等于抹掉了模型最
需要的判别信息。论文的核心论点就是这个：**必须按类别对齐子域，而不是全局对齐**。

LMMD 的定义（论文 Eq.5）：

```
d_H(p, q) = E_c ‖ E_{p(c)} φ(x^s) − E_{q(c)} φ(x^t) ‖²_H
```

离散估计（Eq.8）：

```
d̂_H(p, q) = (1/C) Σ_c ‖ (1/n_s) Σ_i w_i^{sc} φ(x_i^s) − (1/n_t) Σ_j w_j^{tc} φ(x_j^t) ‖²_H
```

权重归一化（Eq.7）：

```
w_i^c = y_i^c / Σ_{(x_j,y_j)∈D} y_j^c
```

- **源域** `w_i^{sc}`：用真标签的 one-hot，`y_i^c ∈ {0,1}`。
- **目标域** `w_j^{tc}`：**用当前模型的软预测概率** `ŷ_j = f(x_j)`，不是硬标签。

目标域用软概率而不是硬伪标签，正是论文为了缓解伪标签错误（负迁移）做的设计——
低置信度的样本只贡献很小的权重，不会像硬伪标签那样被当成确定标签去强化错误。

总损失（Eq.11）：

```
min_f  (1/n_s) Σ_i J(f(x_i^s), y_i^s) + λ Σ d̂(...) + γ L_en
```

本次实现省略第三项 `γ L_en`（熵最小化），即模块③。

## 3. 实现细节

### 3.1 文件

| 文件 | 说明 |
|---|---|
| `ml/src/train_lmmd.py` | 训练脚本，约 780 行，自包含 |
| `kaggle/lmmd/run.py` | Kaggle kernel 驱动，跑 λ 扫描 |
| `kaggle/lmmd/kernel-metadata.json` | kernel 元数据 |

**没有修改** `train_colab.py` / `eval_crossdomain.py` / `data.py` / `models.py`。
`train_lmmd.py` 只 `import` 后两者（`build_model`、`PlantDocDataset`、
`PLANTDOC_TO_PV_FOLDER`、`evaluate`），复用它们作为评估口径的唯一真源，
避免复制一份映射表导致两边漂移。

### 3.2 用核方法，不用显式特征映射

论文 Eq.9 给了核形式，实现直接按它来：

```
d̂_l(p,q) = (1/C) Σ_c [ 1/n_sc² Σ_i Σ_j w_i^{sc} w_j^{sc} k(M_i^s, M_j^s)
                      + 1/n_tc² Σ_i Σ_j w_i^{tc} w_j^{tc} k(M_i^t, M_j^t)
                      − 2/(n_sc n_tc) Σ_i Σ_j w_i^{sc} w_j^{tc} k(M_i^s, M_j^t) ]
```

写成矩阵形式，每个类别 c 只算三个二次型：

```python
term = ws_c @ k_ss @ ws_c + wt_c @ k_tt @ wt_c - 2.0 * (ws_c @ k_st @ wt_c)
```

`k_ss` / `k_tt` / `k_st` 三个核矩阵每个 batch 只算一次（与类别无关），
循环只发生在类别维度上，C=38 也很快。

RBF 核：

```
k(a,b) = exp(−‖a−b‖² / (2σ²))
```

σ² 的取法：`lmmd_sigma=0` 时自动估计——把源域和目标域特征拼起来，算
pairwise 距离矩阵，**去掉对角线**后取中位数。这是 MMD 的标准做法（median heuristic）。
实测 σ² 落在 450~530（resnet18 512 维特征）。

也可以传 `--lmmd-kernel linear` 用线性核 `k(a,b) = aᵀb`，等价于在原始特征空间算
欧氏距离（自测里用它验证过与显式公式一致）。

### 3.3 特征提取：forward_pre_hook

φ 取**分类头之前那层**的输出，用 `register_forward_pre_hook` 抓：

```python
class FeatureTap:
    def __init__(self, model, head_name):
        self.head = model.get_submodule(head_name)
        self.handle = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._feat = inputs[0].flatten(1)   # ConvNeXt 输出是 (B,C,1,1)
```

用 pre-hook 而不是直接 `model.backbone(x)`，好处是对九种模型架构**统一处理**，
不需要为每种模型写一条特征提取路径。`HEAD_MODULE` 映射：

| 模型 | 分类头 |
|---|---|
| resnet18 / resnet50 / shufflenet_v2_x1_0 | `fc` |
| mobilenet_v3_large / small / efficientnet_b0 / convnext_tiny | `classifier` |
| vit_b_16 | `heads` |
| swin_t | `head` |

启动时会先跑一次空前向确认 hook 能抓到 φ 并打印维度，抓不到就报
`没抓到分类头输入特征，检查 HEAD_MODULE 映射是否与模型匹配`，不会静默失败。

### 3.4 双数据流

源域 39170 张、目标域 2670 张，差 15 倍。用 `infinite(target_loader)` 让目标域
loader 循环取，每步同时拿一个源 batch 和一个目标 batch：

```python
for bi, (xs, ys) in enumerate(src_loader):
    xt, _ = next(tgt_iter)          # 目标域标签丢弃
```

`--target-batch-size 0` 表示与源域 batch 同尺寸。

### 3.5 梯度不穿过软概率

`wt = softmax(logits_t).detach()` 是默认行为（`--lmmd-grad-target-probs` 可开）。

理由：如果让梯度穿过 softmax，优化器会发现「把目标域预测全部推成均匀分布」
可以让 LMMD 变小（均匀分布的加权均值更接近源域均值），这是个退化解，会直接
毁掉判别性。detach 后 LMMD 只通过特征 φ 回传梯度，权重被当作常量。

### 3.6 λ 的量级（重要，与原始计划的建议值不同）

任务原本建议 λ 从 0.1 / 0.5 / 1.0 试。**实测发现这个范围基本无效。**

用 `ml/tmp/lmmd_lambda_probe.py`（resnet18 预训练 + 真实数据 6 个 batch，
只统计 `fc` 之外骨干参数的梯度范数）：

| batch | CE | LMMD | ‖g_CE‖ | ‖g_LMMD‖ | 比值 |
|---|---|---|---|---|---|
| 0 | 4.0324 | 0.0981 | 7.748e+00 | 2.886e-01 | 26.8 |
| 1 | 3.6180 | 0.0767 | 8.329e+00 | 3.035e-01 | 27.4 |
| 2 | 3.6252 | 0.0971 | 7.860e+00 | 3.299e-01 | 23.8 |
| 3 | 4.1188 | 0.0855 | 6.998e+00 | 3.309e-01 | 21.1 |
| 4 | 3.7329 | 0.0912 | 7.705e+00 | 3.016e-01 | 25.5 |
| 5 | 3.7216 | 0.1047 | 7.627e+00 | 3.377e-01 | 22.6 |

**梯度比中位数 25.54。** 也就是说：

- λ=0.1 → LMMD 梯度只占 CE 的 **0.4%**，等于没开；
- λ=1.0 → 占 **4%**，还是很弱；
- λ=5.0 → 占 **20%**；
- λ=10.0 → 占 **40%**；
- 要让两项梯度同量级，需要 λ≈25。

所以 `lmmd_weight` 默认值定为 **1.0**（保守起点），Kaggle kernel 扫
**λ ∈ {0.0, 1.0, 5.0, 10.0}**。其中 λ=0.0 是「无 LMMD 但照样加载目标域数据」的对照，
用来分离「LMMD 的贡献」和「多看了 2670 张目标域图片的贡献」——后者本身可能是 0。

如果坚持要照原始建议跑 0.1/0.5/1.0，也能跑，但要预期三条曲线差异极小，
不能据此得出「LMMD 无效」的结论——那是剂量不足，不是方法不行。

### 3.7 其他超参

沿用 `train_colab.py` 的设定以保持可比：

- 优化器 AdamW，骨干 lr = 3e-4 × 0.1，头 lr = 3e-4，weight_decay=0.05
- LambdaLR：线性 warmup + 余弦退火到 0.01 底
- label_smoothing=0.1，`clip_grad_norm_(1.0)`
- 验证/评估口径与 `train_colab.py` 逐字节一致：
  `Resize(int(img_size*256/224)) + CenterCrop + ToTensor + Normalize`

LMMD 相关新增项：

- `--lmmd-warmup-epochs 1`：λ 从 0 线性升到目标值，避免训练初期特征还很乱时
  就被强行对齐。
- LMMD 在 `autocast(enabled=False)` 下用 float32 算。核矩阵 exp 在 fp16 下容易
  出 NaN，这个坑必须避开。
- 非有限损失（NaN/Inf）时跳过该 batch 并告警，不污染权重。

### 3.8 产物

`ml/runs/<run_name>/`：

- `best_source.pt` — 按源域 val acc 选（与历史基线可比）
- `best_target.pt` — 按目标域 top1 选（会偏乐观，只作参考）
- `last.pt`、`classes.json`、`config.json`、`summary.json`
- `history.csv` — 列：`epoch, lambda, train_ce, train_lmmd, train_total, train_acc,
  val_loss, val_acc, target_top1, target_top3, sigma2_mean, target_mass_mean, lr, secs`

## 4. 正确性自测

`ml/tmp/lmmd_selftest.py`，全部断言通过：

| 检查项 | 结果 |
|---|---|
| linear 核 vs 显式加权均值公式 | 19.70401382 vs 19.70401077，差 **3.05e-06** |
| rbf 核 vs 显式三项展开 | 0.87072694 vs 0.87072700，差 **5.96e-08** |
| 目标域近似 one-hot + 特征相同 | lmmd = **0.000e+00** |
| 整体平移 3.0（D=16） | lmmd = **144.0000**，等于 9×16，精确成立 |
| 梯度回传 | \|∂L/∂φ_s\|=2.221e-02，\|∂L/∂φ_t\|=1.609e-02，∂L/∂logits_t = **None**（确认 detach 生效） |
| 部分类别出现（3/5） | lmmd=0.410764，只在出现的类别上平均 |
| 含零向量样本 | lmmd=0.442271，finite=True，不崩 |

一个曾经写错的断言：我最初以为「源域与目标域同分布时 LMMD 应该为 0」。
这是错的——源域用硬 one-hot、目标域用软概率，两边权重本来就不等，同分布也
不会为 0。已改为与显式公式逐项对比，那才是正确的检验方式。

## 5. 冒烟结果（CPU）

### 5.1 第 1 次：从头训（无预训练）

```
python ml/src/train_lmmd.py --epochs 1 --limit-batches 20 --batch-size 8 \
  --no-pretrained --target-eval-limit 2 --run-name smoke_lmmd
```

耗时 328s。φ 维度 512，骨干 60 个张量 lr=3.00e-05，头 2 个张量 lr=3.00e-04。

```
epoch,lambda,train_ce,train_lmmd,train_total,train_acc,val_loss,val_acc,target_top1,target_top3,sigma2_mean,target_mass_mean,lr,secs
0,0.5,3.63113,0.095023,3.678642,0.05,3.509748,0.075006,0.037037,0.074074,54.926521,0.23301,0.00030000,327.9
```

### 5.2 第 2 次：ImageNet 预训练

```
--limit-batches 40 --batch-size 16 --target-eval-limit 2 --run-name smoke_lmmd_pretrained
```

耗时 353s。日志里 σ²≈490~525、每 batch 出现 9~14 个类别、
CE 在 3.28~3.87 波动、LMMD 在 0.095~0.128 波动、train acc 从 0 升到 0.0682。

```
0,0.5,3.569476,0.107408,3.62318,0.095312,3.075129,0.269098,0.055556,0.148148,508.98375,0.469486,0.00030000,353.3
```

### 5.3 第 3 次：3 轮小规模，λ=5.0（唯一能看出跨轮趋势的一次）

```
python ml/src/train_lmmd.py --epochs 3 --limit-batches 20 --batch-size 8 \
  --target-batch-size 8 --num-workers 0 --target-eval-limit 1 \
  --lmmd-weight 5.0 --run-name smoke_lmmd_3ep
```

总耗时 1057s（约 18 分钟）。`ml/runs/smoke_lmmd_3ep/history.csv`：

```
epoch,lambda,train_ce,train_lmmd,train_total,train_acc,val_loss,val_acc,target_top1,target_top3,sigma2_mean,target_mass_mean,lr,secs
0,5.0,3.666771,0.069943,4.016485,0.05625,3.514275,0.122086,0.037037,0.111111,466.556306,0.23416,0.00030000,267.4
1,5.0,3.341339,0.067523,3.678952,0.1875,3.181984,0.257327,0.0,0.111111,463.392801,0.30637,0.00015150,510.9
2,5.0,3.108285,0.059175,3.404161,0.2625,3.109258,0.270021,0.037037,0.074074,457.584259,0.362087,0.00000300,279.0
```

趋势（这是冒烟最有价值的部分）：

| 指标 | ep1 | ep2 | ep3 | 走向 |
|---|---|---|---|---|
| train_ce | 3.6668 | 3.3413 | 3.1083 | ↓ 单调下降 |
| train_lmmd | 0.0699 | 0.0675 | 0.0592 | ↓ 单调下降 |
| train_acc | 0.0563 | 0.1875 | 0.2625 | ↑ 单调上升 |
| val_acc | 0.1221 | 0.2573 | 0.2700 | ↑ |
| σ² 均值 | 466.6 | 463.4 | 457.6 | 稳定 |
| target_mass_mean | 0.234 | 0.306 | 0.362 | ↑ 模型对目标域逐渐有信心 |

**CE 和 LMMD 同时在下降，且 LMMD 的降幅（−15.4%）与 CE 的降幅（−15.2%）几乎
同步**。这排除了「LMMD 只是被 CE 拉着被动变化」和「LMMD 与 CE 互相冲突」两种
失效模式，说明两项损失方向一致。

`target_top1` 在 0.037 / 0.0 / 0.037 之间跳——目标域评估集只有 27 张
（`--target-eval-limit 1` 每类限 1 张），单张变动就是 3.7pp，**没有参考价值**。
要判断 LMMD 是否真的提升跨域精度，必须用全量 252 张评估集，在 Kaggle 上跑。

### 5.4 冒烟结论

- LMMD 能算出来，**无 NaN / Inf**，梯度正常回传，损失有限。
- 产物齐全（best_source.pt / best_target.pt / last.pt / history.csv / summary.json）。
- `target_mass_mean` 0.233（无预训练）→ 0.469（预训练）→ 0.362（3 轮 λ=5）；
  预训练模型对目标域更有信心，软权重更集中，符合预期。
- **CE 与 LMMD 同步下降**，是本次冒烟最有意义的信号。
- **局限**：3 轮、每轮只 20 个 batch（真实一轮是 4896 步），val_acc 只到 0.27
  （完整 15 轮能到 0.99）。这只能证明「能跑、在学、两项损失不打架」，
  **不能证明「有效」**。有效性必须靠 Kaggle 全量 15 轮 + λ=0 对照判断。

## 6. 风险评估

### 6.1 核心风险：伪标签在我们这里已经失败过一次（E-13）

这是**最大的风险，必须写在最前面**。

E-13 测试时适配做过伪标签自训练（`ml/artifacts/tta_adapt.md`）：

| 模型 | 方法 | top1 | 相对基线 |
|---|---|---|---|
| resnet18 | 零样本 baseline | **0.3373** | — |
| resnet18 | 伪标签自训练（99 张伪标签，阈值 0.7） | **0.3135** | **−2.38pp** |
| mobilenet_v3_large | 伪标签自训练（80 张伪标签） | **0.2659** | **−5.56pp** |

当时的分析结论是：**「伪标签的信噪比太低，自训练是拿错标签强化错误」**。

LMMD 用的目标域权重 `w_j^{tc}` 虽然从硬标签换成了软概率，但**信息来源是同一个**：
模型的当前预测。如果模型在 PlantDoc 上本来就只有 28% 准确率，那么软概率里
80% 以上的质量都压在错误类别上，按这些权重对齐就是在**把源域特征往错误的方向推**。

软概率只缓解、不消除这个问题。它降低低置信样本的权重，但高置信度错误（E-09
发现过 311 例高置信度错误）反而会被**放大**——这是最危险的失效模式。

**这一点的严重性怎么强调都不过分**：λ=0 对照组如果和 λ>0 差不多甚至更好，
就说明 LMMD 在这份数据上不成立，应该直接停掉这条路线，不要再花算力试别的 λ。

### 6.2 其他风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 目标域样本太少 | 2670 张 vs 源域 39170 张，每 batch 只有 64 个目标样本覆盖 38 类，平均每类 1.7 个 | 靠软权重累积；但覆盖不足是硬伤 |
| 类别覆盖不足 | 目标域只有 28 类，源域 38 类；每个 batch 实际只出现 7~14 类 | 实现里只对 `present` 类别求和，但对齐项仍除以全部 C，相当于稀释了未出现类别的权重 |
| 类别定义错位 | PlantVillage 38 类 vs PlantDoc 28 类，映射由 `PLANTDOC_TO_PV_FOLDER` 提供，其中 `Tomato_two_spotted_spider_mites_leaf` 被剔除 | 训练时目标域标签丢弃，只有评估才用映射；但如果两个域的类语义有偏移（如 PlantDoc 的「番茄早疫病」样本实际包含晚期才表现的病害），对齐目标本身就偏 |
| σ² 选择敏感 | median heuristic 在两类特征分布差异极大时可能给出过大的 σ²（实测 490~525），导致核矩阵接近全 1，区分度低 | 提供 `--lmmd-sigma` 手动指定；可对比 fixed σ² |
| 计算开销 | 每 batch 多三个核矩阵（64×64），开销可忽略 | — |
| 对齐与 CE 冲突 | 源域 val acc 可能下降，即牺牲判别性换域不变性 | 同时记录 `best_source_val_acc`，看是否明显掉 |

### 6.3 成功概率与预期增益（我的判断）

**成功率我估 25~35%。** 理由：

- 支持：LMMD 比伪标签自训练更温和（软权重 + 只在特征层对齐，不改分类头），
  论文在同类农业跨域任务上报告过有效结果；且它解决的是「全局对齐混簇」这个
  真实存在的问题，不是单纯换个损失。
- 反对：**E-13 的失败机制没有被消除**，只是被削弱。我们的目标域基线只有 28%，
  论文的实验条件（如 Office-31、VisDA）源域准确率通常在 80%+，软概率质量高得多。
  低准确率下的软概率权重，本质上是在放大错误。

**预期增益**：如果有效，我估 **+2~5pp**（28.18% → 30~33%），能达到或略超零样本
最佳的 37.50% 有困难——那需要 +9pp 以上，单靠 LMMD 不太可能。

**最可能的结果**：λ 扫描四条线互相差异在 ±1.5pp 内，全部落在噪声区间。
这种情况下不能说「LMMD 无效」，只能说「在我们的数据规模/基线准确率下测不出效果」。
要在报告里写清楚这个区分。

**建议的判读流程**：

1. 先看 λ=0 对照组。如果 λ=0 就已经明显高于历史基线 28.18%，说明「多加载目标域
   数据但不算 LMMD」本身就有作用（可能是 BN 统计量被目标域影响），后面的比较
   必须减掉这一项。
2. 再看 λ>0 是否稳定优于 λ=0。单次结果在 ±1.5pp 内不可信（本项目历史波动量级）。
3. 如果 λ>0 有效，检查源域 val acc 掉了多少。掉超过 3pp 说明是拿判别性换的。
4. 如果全部 λ>0 都不如 λ=0，**停掉这条路线**，不要再试其他 λ 或 kernel。

## 7. Kaggle 训练配置

### 7.1 文件

- `kaggle/lmmd/kernel-metadata.json`：`id: wangkaiyu66/plant-leaf-lmmd`、
  `enable_gpu: true`、`enable_internet: true`、`machine_shape: NvidiaTeslaT4`
- `kaggle/lmmd/run.py`：驱动脚本

### 7.2 kernel 做什么

1. 从 GitHub raw 拉三个源文件（`train_lmmd.py` / `models.py` / `eval_crossdomain.py`）
2. `pip install datasets`
3. 用 `train_colab.py --prepare-only` 把 PlantVillage 落盘（复用现有下载逻辑）
4. 下载 PlantDoc.zip（HF 主源 + hf-mirror 备源，urllib + Range 续传）
5. 循环 λ ∈ {0.0, 1.0, 5.0, 10.0} 各训 15 轮 resnet18
6. 打印 `ALL_RESULTS_JSON_BEGIN/END` 包裹的汇总，存 `lambda_sweep.json`

### 7.3 预期耗时与配额

4 个配置 × 15 轮 resnet18 @ 224 / T4。粗估每配置 25~40 分钟（含数据加载），
总计 **2~3 小时**。Kaggle 单 kernel 上限 12 小时（GPU 配额 30h/周），够用。

### 7.4 怎么跑（本次不执行）

```powershell
# 先推上去（需要 kaggle.json 已配置）
python -m kaggle kernels push -p kaggle/lmmd

# 看状态
python -m kaggle kernels status wangkaiyu66/plant-leaf-lmmd

# 拉日志（结果 JSON 在日志里）
python -m kaggle kernels output wangkaiyu66/plant-leaf-lmmd -p ml/runs/kaggle_lmmd
```

**注意**：`run.py` 里 `LAMBDAS` 可以改。如果只想先花 1 小时验证方向，
把它改成 `[0.0, 5.0]` 两个配置即可。

## 8. 未做的事（明确边界）

- 没跑全量训练（本机 CPU 太慢，一次 15 轮 resnet18 在本机估计 20+ 小时）。
- 没做 MSUN 的模块①（多表示模块）和模块③（熵最小化）。
- 没做多模型（swin_t / mobilenet）上的 LMMD，只准备了 resnet18 的配置
  （可通过环境变量 `LMMD_MODEL` 切换）。
- 没做 λ 之外的其他超参搜索（σ²、warmup、kernel 类型都保持默认）。
- 没有 git add / commit（按要求由 lead 统一提交）。
