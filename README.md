# 植物叶片病害图像识别分析系统

> 《软件工程原理与实践》课程大作业 · 神经网络方向
> 团队：王楷宇 等 3 人

## 这个项目在做什么

一个面向种植户/农技人员的**叶片病害辅助初筛**系统：手机拍一张叶子，App 返回最可能的病害类别与置信度，并明确提示"仅供参考，不构成诊断"。

技术上它要回答一个具体问题：

> **在叶片病害识别这个任务上，模型的结构和容量如何影响「同域精度」「跨域鲁棒性」「部署成本」这三个互相冲突的目标？手机上到底该用哪种模型？**

不是刷精度。PlantVillage 上 ResNet18 都能到 98%，六模型对比精度差异可能只有 1 个百分点——那个数字本身没有信息量。真正有信息量的是：把模型扔到真实田间照片（PlantDoc）上，精度会掉到多少；以及为了那 1 个百分点要多付多少算力。

## 目录结构

```
.
├── ml/                     机器学习部分
│   ├── src/
│   │   ├── prepare_data.py    parquet → ImageFolder，分层切分
│   │   ├── data.py            数据增强与 DataLoader
│   │   ├── models.py          六模型工厂（统一接口 + 分类头替换）
│   │   ├── train.py           训练循环（分层 lr + warmup + cosine）
│   │   └── bench_cpu.py       CPU 推理速度基准
│   ├── data/               数据集（.gitignore 排除，需本地生成）
│   └── artifacts/          数据体检报告、图表
├── server/                 后端服务（FastAPI + ONNX Runtime + SQLite）
├── app/                    安卓客户端（Kotlin + Jetpack Compose）
├── report/                 课程报告与图表
├── docs/                   项目文档
│   ├── 进度分工面板.md       ★ 任务认领与进度（从这里开始）
│   ├── 实验指导手册.md       111 个任务的详细做法
│   ├── 需求调研/             问卷、访谈提纲、记录模板（调研过程可追溯）
│   ├── 需求规格说明.md       用户画像、14 条用户故事、MoSCoW、需求追踪矩阵
│   ├── 信息架构与页面设计.md  导航决策、4 页 + 2 弹层的线框与四态、设计 token
│   ├── 方案.md               技术方案与实验设计
│   ├── 计划.md               施工步骤 S0-S9、分工、风险
│   ├── 进度表.md             每周进度记录（课程要求公开）
│   └── 讲解/                 知识点讲解（团队内部补课用）
├── scripts/                辅助脚本
└── 例会记录/                团队会议记录（课程要求公开）
```

## 团队协作从这里开始

新成员先读 **[docs/进度分工面板.md](docs/进度分工面板.md)**：找到自己那一节，认领任务，勾选完成。遇到不会做的，去 [docs/实验指导手册.md](docs/实验指导手册.md) 查对应编号的详细步骤。


## 数据从哪来

**PlantVillage** — 公开学术数据集，38 类、约 5.4 万张叶片图像。

学术引用最多的版本来自 Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science。

本项目实际使用 HuggingFace 镜像 [`GVJahnavi/PlantVillage_dataset`](https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset)，转换后：

| split | 数量 |
|---|---|
| train | 39170 |
| val | 4333（从 train 分层切出 10%，seed=42） |
| test | 10878 |

**跨域测试集**：PlantDoc，用于检验模型在真实拍摄条件下的表现。

> ⚠️ 数据集本身有已知局限：全部是实验室摆拍（单一背景、固定光照、单叶居中），模型容易学到背景而不是病斑。这是本项目的核心实验对象，不是要回避的问题。

## 快速开始

### 1. 环境

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyarrow pandas pillow numpy scikit-learn tqdm matplotlib
```

### 2. 准备数据

从 HuggingFace 下载三个 parquet 到 `ml/data/raw/plantvillage/`，然后：

```bash
cd ml
python src/prepare_data.py --raw-dir data/raw/plantvillage --out-dir data/processed/plantvillage
```

产出 `data/processed/plantvillage/{train,val,test}/<类别>/<图片>.jpg` 和 `classes.json`。

### 3. 训练

```bash
cd ml
python src/train.py --model resnet18 --data-root data/processed/plantvillage --epochs 15
```

产物在 `ml/runs/<run_name>/`：`config.json`、`history.csv`、`best.pt`、`summary.json`、`test_predictions.npz`。

本地 CPU 太慢，正式训练在 Colab 上跑。

### 4. 冒烟测试（验证管线是否通）

```bash
cd ml
python src/train.py --model resnet18 --data-root data/smoke \
    --epochs 1 --limit-batches 3 --batch-size 16 --num-workers 0
```

## 实验设计

| 编号 | 内容 | 状态 |
|---|---|---|
| E1 | 六模型结构对比（ResNet18/50、MobileNetV3、ShuffleNetV2、EfficientNet-B0、Swin-T） | 待跑 |
| E2 | 跨域泛化：PlantVillage 训练 → PlantDoc 测试 | 待跑 |
| E3 | 效率实测：参数量 / FLOPs / CPU 延迟 | 部分完成 |
| E7 | Grad-CAM：同域 vs 跨域时模型到底在看哪 | 待跑 |

## 重要约定

- **不提交数据和大文件**：`ml/data/`（2.37GB）、`*.pt`、`*.onnx` 全部在 `.gitignore` 里。新人 clone 后需按上面步骤自行生成数据。
- **提交实验记录**：`ml/runs/*/summary.json`、`history.csv`、`config.json` 要提交——报告里的每个数字都必须能追溯到某次具体运行。
- **模型必须写明是微调**：全部实验基于 ImageNet 预训练权重。报告里不能含糊成"我们训练的模型"。

## 许可与致谢

- PlantVillage 数据集版权归原作者所有，本项目仅用于课程学习。
- 网络结构与预训练权重来自 [torchvision](https://github.com/pytorch/vision)。
