"""Kaggle Kernel：Swin-T 训练（15 轮，Transformer 对照）

之前 Colab 每次跑到 ep13/ep14 就被 100 分钟上限掐掉，Kaggle 单次 12 小时可以跑完。
产物写到 /kaggle/working/runs，用 `kaggle kernels output` 拉回本地。
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
RUN_NAME = os.environ.get("RUN_NAME", "swin_t_224_15ep")


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def main():
    log("=" * 60)
    log(f"python {sys.version.split()[0]}  cwd={os.getcwd()}")
    log(f"run_name={RUN_NAME}")

    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"gpu={p.name} vram={p.total_memory/1e9:.1f}GB")
    log(f"nproc={os.cpu_count()}")

    log("安装 datasets …")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

    log(f"下载训练脚本 {TRAIN_SCRIPT_URL}")
    os.makedirs(WORK, exist_ok=True)
    script_path = f"{WORK}/train_colab.py"
    urllib.request.urlretrieve(TRAIN_SCRIPT_URL, script_path)
    log(f"脚本大小 {os.path.getsize(script_path)} 字节")

    # Swin-T 显存占用大、速度慢，batch 降到 32，lr 相应保持
    cmd = [
        sys.executable, "-u", script_path,
        "--model", "swin_t",
        "--epochs", "15",
        "--img-size", "224",
        "--batch-size", "32",
        "--lr", "0.0002",
        "--data-root", DATA_ROOT,
        "--out-dir", OUT_DIR,
        "--run-name", RUN_NAME,
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
