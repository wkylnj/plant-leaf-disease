"""在一次性 Colab VM 上训练单个模型，并把产物以 base64 tarball 打到 stdout。

为什么这样设计：
    前两次训练都栽在"会话被回收、/content 全丢"。`colab run` 虽然一次性、
    跑完即释放，但它把 stdout 流回本地 —— 只要训练结束时把产物编码进
    stdout，就算 VM 随后被销毁，结果也已经在我手里。

    只打包小文件（json/csv/png），不含 *.pt 权重（43MB base64 会把 stdout 撑爆）。
    需要权重时另用 colab_fetch.py 或重新训练。

用法（本地）：
    colab run --gpu T4 --timeout 600 scripts/colab_train_one.py swin_t
    colab run --gpu T4 --timeout 600 scripts/colab_train_one.py mobilenet_v3_large --epochs 15

脚本参数：
    model              模型名（必填）
    --epochs           轮数，默认 15
    --img-size         分辨率，默认 224
    --batch-size       batch，默认按模型选
    --lr               学习率，默认按模型选
    --run-name         run 名，默认 <model>_<img_size>_<epochs>ep
    --skip-data        跳过数据准备（VM 内已有 /content/data 时用）
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"
TRAIN_SCRIPT_URL = f"{RAW_BASE}/ml/src/train_colab.py"
TRAIN_SCRIPT_PATH = Path("/content/work/train_colab.py")
DATA_ROOT = "/content/data/plantvillage"
RUNS_DIR = "/content/runs"

MODEL_BATCH = {
    "resnet18": 64,
    "resnet50": 48,
    "mobilenet_v3_large": 64,
    "mobilenet_v3_small": 96,
    "shufflenet_v2_x1_0": 96,
    "efficientnet_b0": 64,
    "swin_t": 32,
    "vit_b_16": 32,
}
MODEL_LR = {
    "resnet18": "3e-4",
    "resnet50": "3e-4",
    "mobilenet_v3_large": "3e-4",
    "mobilenet_v3_small": "3e-4",
    "shufflenet_v2_x1_0": "3e-4",
    "efficientnet_b0": "3e-4",
    "swin_t": "2e-4",
    "vit_b_16": "2e-4",
}

EXCLUDE_SUFFIX = (".pt", ".pth", ".onnx", ".npz", ".npy")


def log(msg: str) -> None:
    print(f"[one] {msg}", flush=True)


def fetch_train_script() -> None:
    TRAIN_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"下载 train_colab.py ← {TRAIN_SCRIPT_URL}")
    with urllib.request.urlopen(TRAIN_SCRIPT_URL, timeout=120) as r:
        data = r.read()
    TRAIN_SCRIPT_PATH.write_bytes(data)
    log(f"已保存 {TRAIN_SCRIPT_PATH}（{len(data)} 字节）")


def run_training(env_extra: dict[str, str], log_path: Path) -> int:
    """跑训练，同时把输出实时透传到 stdout（这样本地能看到进度）。"""
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONUTF8"] = "1"
    log(f"开始训练：{env_extra.get('COLAB_RUN_NAME')}")
    with log_path.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(
            [sys.executable, "-u", str(TRAIN_SCRIPT_PATH)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        p.wait()
    return p.returncode


def pack_run(run_dir: Path) -> tuple[str, int]:
    """把小文件打成 tar.gz 并 base64 编码。"""
    buf = io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name.endswith(EXCLUDE_SUFFIX):
                continue
            tf.add(f, arcname=str(f.relative_to(run_dir)))
            n_files += 1
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), n_files


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("用法: colab_train_one.py <model> [--epochs 15] ...", file=sys.stderr)
        return 2
    model = argv[0]
    rest = argv[1:]

    epochs = 15
    img_size = 224
    batch_size = MODEL_BATCH.get(model, 64)
    lr = MODEL_LR.get(model, "3e-4")
    run_name = None
    skip_data = False

    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--epochs":
            epochs = int(rest[i + 1]); i += 2
        elif a == "--img-size":
            img_size = int(rest[i + 1]); i += 2
        elif a == "--batch-size":
            batch_size = int(rest[i + 1]); i += 2
        elif a == "--lr":
            lr = rest[i + 1]; i += 2
        elif a == "--run-name":
            run_name = rest[i + 1]; i += 2
        elif a == "--skip-data":
            skip_data = True; i += 1
        else:
            print(f"未知参数 {a}", file=sys.stderr)
            return 2

    run_name = run_name or f"{model}_{img_size}_{epochs}ep"
    log(f"model={model} epochs={epochs} img_size={img_size} bs={batch_size} lr={lr}")
    log(f"run_name={run_name}")

    fetch_train_script()

    env_extra = {
        "COLAB_MODEL": model,
        "COLAB_EPOCHS": str(epochs),
        "COLAB_IMG_SIZE": str(img_size),
        "COLAB_BATCH_SIZE": str(batch_size),
        "COLAB_NUM_WORKERS": "2",
        "COLAB_LR": lr,
        "COLAB_STRONG_AUG": "1",
        "COLAB_RUN_NAME": run_name,
        "COLAB_BENCH": "0",
        "COLAB_PREPARE_ONLY": "1" if skip_data else "0",
        "COLAB_DATA_ROOT": DATA_ROOT,
        "COLAB_OUT_DIR": RUNS_DIR,
    }

    rc = run_training(env_extra, Path(f"/content/{run_name}.log"))

    run_dir = Path(RUNS_DIR) / run_name
    has_summary = (run_dir / "summary.json").exists()
    log(f"训练结束 rc={rc} summary_exists={has_summary}")

    if not run_dir.exists():
        log("run 目录不存在，无产物可打包")
        print("===ARTIFACT_END===")
        return 1

    b64, n = pack_run(run_dir)
    log(f"打包 {n} 个文件，base64 {len(b64)} 字符")
    print("===ARTIFACT_BEGIN===")
    print(f"RUN_NAME={run_name}")
    print(f"FILE_COUNT={n}")
    print(f"HAS_SUMMARY={has_summary}")
    print(f"RC={rc}")
    print(b64)
    print("===ARTIFACT_END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
