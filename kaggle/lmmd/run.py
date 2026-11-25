"""Kaggle Kernel：LMMD 子域对齐（MSUN 论文第②模块）

实验目的
--------
验证「按类别对齐子域」能否把 PlantVillage → PlantDoc 的跨域精度推过现有基线。
现有基线：零样本最佳 37.50%（Swin-T 多分辨率平均），ResNet18 28.18%。

本 kernel 只做一个变量：在源域 CE 之外加 LMMD 项。
不做 MSUN 的①多表示模块（Inception Adaptation），也不做③不确定性正则（熵最小化）。
这样如果精度变了，归因是干净的。

λ 取值说明（重要）
------------------
论文 Eq.11 只写了 λ，没给数值。本机用 resnet18 预训练 + 真实数据测过两项的梯度
量级（ml/tmp/lmmd_lambda_probe.py）：CE 在骨干上梯度范数 ≈ 7.7，LMMD ≈ 0.31，
比值 25.5。所以 λ=0.1/0.5/1.0 时 LMMD 的梯度只占 CE 的 0.4%/2%/4%，等于没开。
本 kernel 因此扫 λ ∈ {1.0, 5.0, 10.0}，分别对应 LMMD 梯度约占 CE 的 4%/20%/40%。
如果你想严格照"λ 从 0.1/0.5/1.0 试"的原始计划，把 LAMBDAS 改成 [0.1, 0.5, 1.0]
即可，但要预期三条曲线的差异会非常小。

为什么还要跑 λ=0 的对照
-----------------------
LMMD 依赖目标域的软预测概率作为权重，这和本项目之前失败的伪标签自训练（E-13，
0.3373 → 0.3135）是同一类风险。必须有一个"只训源域、不加载目标域"的干净对照，
否则无法判断精度变化来自 LMMD 还是来自"多看了 2670 张目标域图片"。

Kaggle 环境说明
---------------
- 预装 torch / torchvision，需补装 datasets（下载 PlantVillage 用）
- 联网开启后可从 HuggingFace 下 PlantVillage（约 813MB）和 PlantDoc（约 200MB）
- 产物写到 /kaggle/working/runs，用 `kaggle kernels output` 拉回本地

用法（本地驱动，本次不执行）：
    python -m kaggle kernels push -p kaggle/lmmd --accelerator NvidiaTeslaT4
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"
SRC_FILES = {
    # train_lmmd.py 依赖 models.py（build_model）和 eval_crossdomain.py
    # （PlantDocDataset + PLANTDOC_TO_PV_FOLDER + evaluate，评估口径的唯一真源）
    "train_lmmd.py": f"{RAW_BASE}/ml/src/train_lmmd.py",
    "models.py": f"{RAW_BASE}/ml/src/models.py",
    "eval_crossdomain.py": f"{RAW_BASE}/ml/src/eval_crossdomain.py",
}

WORK = "/kaggle/working"
SRC = f"{WORK}/src"
PV_ROOT = f"{WORK}/data/plantvillage"
PD_ROOT = f"{WORK}/data/plantdoc"
OUT_DIR = f"{WORK}/runs"
PD_ZIP = f"{WORK}/PlantDoc.zip"

PD_MIRRORS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]

# 模型与超参
MODEL = os.environ.get("LMMD_MODEL", "resnet18")
EPOCHS = "15"
BATCH_SIZE = "64"
TARGET_BATCH_SIZE = "64"
LR = "0.0003"
IMG_SIZE = "224"
# λ 扫描表。0.0 是无 LMMD 的对照（但仍加载目标域数据，见下）
LAMBDAS = [0.0, 1.0, 5.0, 10.0]


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest, attempts=3):
    """urllib 下载 + Range 续传。HF 支持 Range，断线不用重下。"""
    for attempt in range(1, attempts + 1):
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        headers = {"User-Agent": "Mozilla/5.0 (Kaggle)"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            log(f"续传：已有 {human(existing)}")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0) + existing
                mode = "ab" if (existing and resp.status == 206) else "wb"
                if mode == "wb":
                    existing = 0
                log(f"下载 {url} → 总大小 {human(total) if total else '未知'}")
                done = existing
                t0 = last = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last >= 5:
                            speed = (done - existing) / max(now - t0, 1e-6)
                            pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                            log(f"  {pct} {human(done)} / {human(total) if total else '?'}  "
                                f"{human(speed)}/s")
                            last = now
            size = os.path.getsize(dest)
            if total and size < total:
                raise IOError(f"未下完：{human(size)} / {human(total)}")
            log(f"下载完成：{human(size)}")
            return True
        except Exception as e:
            log(f"下载失败（第 {attempt} 次）：{type(e).__name__}: {e}")
            time.sleep(5)
    return False


def fetch_scripts():
    os.makedirs(SRC, exist_ok=True)
    for name, url in SRC_FILES.items():
        path = os.path.join(SRC, name)
        log(f"拉取 {name}")
        urllib.request.urlretrieve(url, path)
        log(f"  {os.path.getsize(path)} 字节")


def fetch_plantdoc():
    if os.path.isdir(f"{PD_ROOT}/train") and os.path.isdir(f"{PD_ROOT}/test"):
        log("PlantDoc 已解压，跳过")
        return True
    for url in PD_MIRRORS:
        if download(url, PD_ZIP):
            break
    else:
        log("所有镜像都失败")
        return False
    log("校验并解压 …")
    try:
        with zipfile.ZipFile(PD_ZIP) as z:
            bad = z.testzip()
            if bad:
                raise IOError(f"zip 损坏于 {bad}")
            z.extractall(PD_ROOT)
    except Exception as e:
        log(f"解压失败：{type(e).__name__}: {e}")
        return False
    ok = os.path.isdir(f"{PD_ROOT}/train") and os.path.isdir(f"{PD_ROOT}/test")
    if ok:
        n_tr = sum(len(f) for _, _, f in os.walk(f"{PD_ROOT}/train"))
        n_te = sum(len(f) for _, _, f in os.walk(f"{PD_ROOT}/test"))
        log(f"PlantDoc 图片数：train={n_tr} test={n_te}（期望 2670 / 252）")
    return ok


def run_one(lam, common_args):
    """跑一个 λ 配置，返回从 summary.json 读到的结果。"""
    tag = f"lam{lam:g}"
    run_name = f"{MODEL}_lmmd_{tag}_224_{EPOCHS}ep"
    cmd = [
        sys.executable, "-u", f"{SRC}/train_lmmd.py",
        "--model", MODEL,
        "--epochs", EPOCHS,
        "--img-size", IMG_SIZE,
        "--batch-size", BATCH_SIZE,
        "--target-batch-size", TARGET_BATCH_SIZE,
        "--lr", LR,
        "--lmmd-weight", f"{lam:g}",
        "--data-root", PV_ROOT,
        "--target-root", PD_ROOT,
        "--out-dir", OUT_DIR,
        "--run-name", run_name,
        "--num-workers", "2",
    ] + common_args
    log(f"[λ={lam:g}] 开始：{' '.join(cmd)}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc = subprocess.call(cmd, cwd=SRC, env=env)
    secs = time.time() - t0
    log(f"[λ={lam:g}] 退出码 {rc}，耗时 {secs / 60:.1f} 分钟")

    summary_path = f"{OUT_DIR}/{run_name}/summary.json"
    if rc != 0 or not os.path.exists(summary_path):
        return {"lambda": lam, "error": f"rc={rc}"}
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    return {
        "lambda": lam,
        "run_name": run_name,
        "best_source_val_acc": s.get("best_source_val_acc"),
        "best_target_top1": s.get("best_target_top1"),
        "final_target_top1": s.get("final_target_top1"),
        "final_target_top3": s.get("final_target_top3"),
        "secs": round(secs, 1),
    }


def main():
    log("=" * 70)
    log(f"python {sys.version.split()[0]}")
    log(f"cwd={os.getcwd()}")
    log(f"model={MODEL} lambdas={LAMBDAS}")

    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"gpu={p.name} vram={p.total_memory / 1e9:.1f}GB")
    log(f"nproc={os.cpu_count()}")
    st = os.statvfs(WORK)
    log(f"/kaggle/working 可用 {st.f_bavail * st.f_frsize / 1e9:.1f}GB")

    os.makedirs(OUT_DIR, exist_ok=True)
    fetch_scripts()

    # 先补装 datasets（train_lmmd.py 的 download_and_prepare 需要它下载 PlantVillage）
    log("安装 datasets …")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

    # PlantVillage：train_lmmd.py 只读本地目录，不做下载。这里借用
    # train_colab.py 的 --prepare-only 模式先把 38 类目录落盘。
    log("准备 PlantVillage（借用 train_colab.py 的 prepare_only 模式）…")
    prep_script = f"{SRC}/train_colab.py"
    if not os.path.exists(prep_script):
        urllib.request.urlretrieve(f"{RAW_BASE}/ml/src/train_colab.py", prep_script)
    rc = subprocess.call([
        sys.executable, "-u", prep_script,
        "--prepare-only",
        "--data-root", PV_ROOT,
    ], cwd=SRC, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log(f"PlantVillage 准备退出码 {rc}")
    if rc != 0:
        log("PlantVillage 准备失败，终止")
        sys.exit(2)

    if not fetch_plantdoc():
        log("PlantDoc 不可用，终止")
        sys.exit(2)

    # ---- λ 扫描 ----
    # 不传 --export-onnx：本实验只看精度，导出 ONNX 白花时间
    common = ["--strong-aug"]
    results = []
    for lam in LAMBDAS:
        results.append(run_one(lam, common))

    # 结果打到 stdout，本地可以直接从 kernel 日志抓，不必先拉 output
    print("\n" + "=" * 70)
    print("ALL_RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print("ALL_RESULTS_JSON_END", flush=True)

    summary_path = f"{OUT_DIR}/lambda_sweep.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"已保存 {summary_path}")

    # 汇总表
    print("\n[λ 扫描结果]")
    print(f"{'λ':>6}{'源域 val acc':>16}{'目标域 top-1':>16}{'目标域 top-3':>16}{'分钟':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['lambda']:>6}{'失败: ' + r['error']:>16}")
            continue
        print(f"{r['lambda']:>6}"
              f"{(r['best_source_val_acc'] or 0):>16.4f}"
              f"{(r['best_target_top1'] or 0):>16.4f}"
              f"{(r['final_target_top3'] or 0):>16.4f}"
              f"{r['secs'] / 60:>10.1f}")

    print("\n判读要点：")
    print("1. λ=0 是无 LMMD 对照（但仍加载目标域数据）。LMMD 是否有效，看 λ>0 相对 λ=0。")
    print("2. 若所有 λ>0 都不如 λ=0，说明按软概率加权的对齐在本数据上是负迁移——")
    print("   与之前伪标签自训练 E-13 失败的机制一致，应立即停止这条路线。")
    print("3. 若某个 λ 有效，检查该 λ 下源域 val acc 是否明显掉。掉很多说明对齐")
    print("   是以牺牲源域判别性换来的，泛化收益可能不可持续。")

    if all("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
