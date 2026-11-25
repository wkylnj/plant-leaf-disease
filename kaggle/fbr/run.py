"""Kaggle Kernel：FBR（Field Background Recomposition）背景重组训练。

方法（Jeon et al., Ecological Informatics 2026）：
    把 PlantVillage 的实验室摆拍叶片抠出来，贴到 PlantDoc 的真实田间背景上，
    让模型训练时就见到田间成像风格。论文报告仅这一步 +9.9pp。

与基线唯一的不同：
    `--data-root` 指向 FBR 合成图，其余超参（resnet18 / 15ep / bs64 / lr3e-4）
    与 runs/resnet18_224_15ep 完全一致，保证是单变量对比。

Kaggle 环境说明：
- 预装 torch / torchvision / scipy，需补装 datasets（下载 PlantVillage 用）
- 联网开启后可直接从 HuggingFace 下载 PlantVillage（约 813MB）
- PlantDoc 从 HF 下载（LamTNguyen/PlantDoc，约 900MB），官方源失败自动换 hf-mirror
- 产物写到 /kaggle/working/runs，用 `kaggle kernels output` 拉回本地

数据流：
    HF PlantVillage ──train_colab.download_and_prepare──> /kaggle/working/data/plantvillage
                                                              │
    HF PlantDoc ──make_bg_library.py──> data/raw/plantdoc_bg ──┤
                                                              ↓
                                        make_fbr_dataset.py（合成）
                                                              ↓
                                              data/processed/pv_fbr（扁平 <类>/*.jpg）
                                                              ↓
                                        prepare_fbr_marker.py（切 train/val + .prepared）
                                                              ↓
                                        train_colab.py --data-root .../pv_fbr

为什么不改 train_colab.py：
    它判断「数据是否就绪」只看 `data_root/.prepared` 是否存在。所以只要
    prepare_fbr_marker.py 把扁平合成图组织成 train/val/test 并写好标记，
    训练脚本就完全跳过下载，FBR 与基线走同一条代码路径，可比性最强。

用法（本地驱动，注意不要在本机执行 push）：
    python -m kaggle kernels push -p kaggle\\fbr --accelerator NvidiaTeslaT4
"""

import os
import subprocess
import sys
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"

# 需要拉取的脚本。make_bg_library.py 内部 `from eval_leafcrop import leaf_mask`，
# 而 eval_leafcrop 又 import data / models / eval_crossdomain —— 这几个都得一起下，
# 否则 Kaggle 上直接 ImportError。
SCRIPTS = {
    "train_colab.py": f"{RAW_BASE}/ml/src/train_colab.py",
    "make_bg_library.py": f"{RAW_BASE}/ml/src/make_bg_library.py",
    "make_fbr_dataset.py": f"{RAW_BASE}/ml/src/make_fbr_dataset.py",
    "prepare_fbr_marker.py": f"{RAW_BASE}/ml/src/prepare_fbr_marker.py",
    # make_bg_library → eval_leafcrop.leaf_mask 的依赖链
    "eval_leafcrop.py": f"{RAW_BASE}/ml/src/eval_leafcrop.py",
    "eval_crossdomain.py": f"{RAW_BASE}/ml/src/eval_crossdomain.py",
    "data.py": f"{RAW_BASE}/ml/src/data.py",
    "models.py": f"{RAW_BASE}/ml/src/models.py",
}

PLANTDOC_URLS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]

WORK = "/kaggle/working"
SRC = f"{WORK}/src"                      # 脚本统一放这，模拟 ml/src 的扁平 import
PV_ROOT = f"{WORK}/data/plantvillage"    # train_colab 下载的原始 PlantVillage
PLANTDOC_ROOT = f"{WORK}/data/raw/plantdoc"
BG_ROOT = f"{WORK}/data/raw/plantdoc_bg"
FBR_ROOT = f"{WORK}/data/processed/pv_fbr"
OUT_DIR = f"{WORK}/runs"

# 与 runs/resnet18_224_15ep 完全一致，只换 data-root
MODEL = os.environ.get("FBR_MODEL", "resnet18")
EPOCHS = os.environ.get("FBR_EPOCHS", "15")
PER_IMAGE = os.environ.get("FBR_PER_IMAGE", "1")
RUN_NAME = os.environ.get("RUN_NAME", f"{MODEL}_fbr_224_{EPOCHS}ep")


def log(msg):
    print(f"[kaggle-fbr] {msg}", flush=True)


def fetch_scripts():
    os.makedirs(SRC, exist_ok=True)
    for name, url in SCRIPTS.items():
        dst = os.path.join(SRC, name)
        log(f"下载 {name}")
        urllib.request.urlretrieve(url, dst)
        log(f"  {os.path.getsize(dst)} 字节")
    return SRC


def ensure_plantvillage():
    """用 train_colab.py 自己的 download_and_prepare 下 PlantVillage。

    这样做而不是另写一份下载逻辑：保证 Kaggle 上的 PlantVillage 目录结构
    与本地 / Colab 完全一致（含 classes.json 与 .prepared）。
    """
    marker = os.path.join(PV_ROOT, ".prepared")
    if os.path.exists(marker):
        log(f"PlantVillage 已存在: {PV_ROOT}")
        return

    log("PlantVillage 不存在，调用 train_colab.prepare_only 下载并转换…")
    sys.path.insert(0, SRC)
    cmd = [
        sys.executable, "-u", os.path.join(SRC, "train_colab.py"),
        "--data-root", PV_ROOT,
        "--prepare-only",
    ]
    log("执行: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=WORK)
    if not os.path.exists(marker):
        raise RuntimeError(f"PlantVillage 准备失败，缺少 {marker}")


def ensure_plantdoc():
    """下载并解压 PlantDoc。已有则跳过。"""
    test_dir = os.path.join(PLANTDOC_ROOT, "test")
    if os.path.isdir(test_dir):
        log(f"PlantDoc 已存在: {PLANTDOC_ROOT}")
        return

    os.makedirs(PLANTDOC_ROOT, exist_ok=True)
    zip_path = os.path.join(PLANTDOC_ROOT, "PlantDoc.zip")

    last_err = None
    for url in PLANTDOC_URLS:
        try:
            log(f"下载 PlantDoc: {url}")
            urllib.request.urlretrieve(url, zip_path)
            log(f"  已下载 {os.path.getsize(zip_path)/1e6:.0f} MB")
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"  失败: {type(e).__name__}: {e}")

    if last_err is not None:
        raise RuntimeError(f"PlantDoc 下载失败: {last_err}")

    import zipfile
    log("解压 PlantDoc…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(PLANTDOC_ROOT)

    # 压缩包可能多套一层目录，找含 test/ 的那层并提上来
    if not os.path.isdir(test_dir):
        for name in os.listdir(PLANTDOC_ROOT):
            cand = os.path.join(PLANTDOC_ROOT, name)
            if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "test")):
                log(f"压缩包多一层目录 {name}/，提上来")
                for sub in os.listdir(cand):
                    os.rename(os.path.join(cand, sub), os.path.join(PLANTDOC_ROOT, sub))
                os.rmdir(cand)
                break

    if not os.path.isdir(test_dir):
        raise RuntimeError(f"解压后仍找不到 {test_dir}")


def build_bg_library():
    os.makedirs(BG_ROOT, exist_ok=True)
    n = len([f for f in os.listdir(BG_ROOT) if f.startswith("bg_") and f.endswith(".jpg")])
    if n >= 100:
        log(f"背景库已存在 {n} 张，跳过重建")
        return
    cmd = [
        sys.executable, "-u", os.path.join(SRC, "make_bg_library.py"),
        "--plantdoc-root", PLANTDOC_ROOT,
        "--out", BG_ROOT,
        "--target", "400",
    ]
    log("执行: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=WORK)


def build_fbr_dataset():
    meta = os.path.join(FBR_ROOT, "fbr_meta.json")
    if os.path.exists(meta):
        log(f"FBR 数据已存在: {FBR_ROOT}")
        return
    cmd = [
        sys.executable, "-u", os.path.join(SRC, "make_fbr_dataset.py"),
        "--pv-root", os.path.join(PV_ROOT, "train"),
        "--bg-root", BG_ROOT,
        "--out", FBR_ROOT,
        "--per-image", PER_IMAGE,
        "--size", "224",
    ]
    log("执行: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=WORK)


def prepare_marker():
    cmd = [
        sys.executable, "-u", os.path.join(SRC, "prepare_fbr_marker.py"),
        "--fbr-root", FBR_ROOT,
        "--pv-root", PV_ROOT,
    ]
    log("执行: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=WORK)


def count_files(root):
    if not os.path.isdir(root):
        return 0
    return sum(len(fs) for _, _, fs in os.walk(root))


def main():
    log("=" * 60)
    log(f"python {sys.version.split()[0]}")
    log(f"run_name={RUN_NAME} model={MODEL} epochs={EPOCHS} per_image={PER_IMAGE}")

    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"gpu={torch.cuda.get_device_name(0)} vram={p.total_memory/1e9:.1f}GB")
    log(f"nproc={os.cpu_count()}")

    # datasets 用于下载 PlantVillage；scipy 用于合成图的形态学操作
    log("安装 datasets / scipy …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "datasets", "scipy"], check=True
    )

    fetch_scripts()
    ensure_plantvillage()
    ensure_plantdoc()
    build_bg_library()
    build_fbr_dataset()
    prepare_marker()

    n_train = count_files(os.path.join(FBR_ROOT, "train"))
    n_val = count_files(os.path.join(FBR_ROOT, "val"))
    log(f"FBR 数据就绪: train={n_train} val={n_val}")

    # 关键自检：确认 .prepared 在，且 train/val 下确实是 _fbr 合成图
    assert os.path.exists(os.path.join(FBR_ROOT, ".prepared")), "缺少 .prepared，训练会触发下载"
    sample = []
    for cls in sorted(os.listdir(os.path.join(FBR_ROOT, "train")))[:3]:
        d = os.path.join(FBR_ROOT, "train", cls)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))[:2]
            sample += [f"{cls}/{f}" for f in files]
    log(f"训练样本抽样（应含 _fbr 后缀）: {sample}")
    if not any("_fbr" in s for s in sample):
        log("警告：抽样里没有 _fbr 后缀，请确认用的是合成图")

    cmd = [
        sys.executable, "-u", os.path.join(SRC, "train_colab.py"),
        "--model", MODEL,
        "--epochs", EPOCHS,
        "--img-size", "224",
        "--batch-size", "64",
        "--lr", "0.0003",
        "--data-root", FBR_ROOT,
        "--out-dir", OUT_DIR,
        "--run-name", RUN_NAME,
        "--export-onnx",
    ]
    log("执行: " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=WORK)
    log(f"训练进程退出码 {rc}")

    run_dir = os.path.join(OUT_DIR, RUN_NAME)
    if os.path.isdir(run_dir):
        log("产物清单:")
        for name in sorted(os.listdir(run_dir)):
            p = os.path.join(run_dir, name)
            if os.path.isfile(p):
                log(f"  {name}  {os.path.getsize(p)/1e6:.2f} MB")
    else:
        log(f"警告：产物目录不存在 {run_dir}")

    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
