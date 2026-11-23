"""Kaggle Kernel：域适应增强训练（MobileNetV3-Large）

Kaggle 环境说明：
- 预装 torch / torchvision，需补装 datasets
- 联网开启后可直接从 HuggingFace 下载 PlantVillage（约 813MB）
- 产物写到 /kaggle/working/runs，用 `kaggle kernels output` 拉回本地

用法（本地驱动）：
    python scripts/kaggle_submit.py da_aug_mobilenet
"""

import os
import subprocess
import sys
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"
TRAIN_SCRIPT_URL = f"{RAW_BASE}/ml/src/train_colab.py"

WORK = "/kaggle/working"
DATA_ROOT = f"{WORK}/data/plantvillage"
OUT_DIR = f"{WORK}/runs"
RUN_NAME = os.environ.get("RUN_NAME", "mobilenet_v3_large_daaug_224_15ep")


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def main():
    log("=" * 60)
    log(f"python {sys.version.split()[0]}")
    log(f"cwd={os.getcwd()}")
    log(f"data_root={DATA_ROOT}")
    log(f"out_dir={OUT_DIR}")
    log(f"run_name={RUN_NAME}")

    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    log(f"nproc={os.cpu_count()}")

    # datasets 是唯一需要补装的依赖
    log("安装 datasets …")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

    # 从 GitHub 拉训练脚本（保持单一真源，不在 Kaggle 里维护副本）
    log(f"下载训练脚本 {TRAIN_SCRIPT_URL}")
    os.makedirs(WORK, exist_ok=True)
    script_path = f"{WORK}/train_colab.py"
    urllib.request.urlretrieve(TRAIN_SCRIPT_URL, script_path)
    log(f"脚本大小 {os.path.getsize(script_path)} 字节")

    cmd = [
        sys.executable, "-u", script_path,
        "--model", "mobilenet_v3_large",
        "--epochs", "15",
        "--img-size", "224",
        "--batch-size", "64",
        "--lr", "0.0003",
        "--data-root", DATA_ROOT,
        "--out-dir", OUT_DIR,
        "--run-name", RUN_NAME,
        "--da-aug",
        "--export-onnx",
    ]
    log("执行: " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=WORK)
    log(f"训练进程退出码 {rc}")

    run_dir = f"{OUT_DIR}/{RUN_NAME}"
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
