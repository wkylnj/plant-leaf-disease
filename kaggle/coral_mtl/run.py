"""Kaggle Kernel：CORAL 特征对齐 + 多任务学习（作物 / 病害）无监督域适应。

背景
----
PlantVillage（源域，38 类有标签）→ PlantDoc（目标域，27 类无标签）的零样本
跨域基线只有 28.18% top-1、51.13% 作物级。已有的六条「不碰权重、只改输入/输出」
路线（da_aug、无标签 TTA、CIELAB 颜色对齐、叶片定位裁剪、多模型集成、局部裁剪
微调）全部证伪，说明掉点主因是**特征本身不对**，不是输入统计量漂移。

本实验直接改特征：训练时同时看两个域，用 CORAL 把目标域特征的协方差拉向源域，
同时用作物分类作为辅助任务，强制共享特征保留域无关的粗粒度语义轴。

关键设计（详见 ml/artifacts/coral_mtl.md）
------------------------------------------
1. 双头：病害头 38 类（主任务）+ 作物头 14 类（辅助），总损失
   L = CE_病害 + λ·CE_作物 + μ·L_coral，λ 默认 0.4。
2. CORAL 只对齐二阶统计量：L = ||Cov(S) − Cov(T)||²_F / (4d²)。
3. 目标域标签读出来即丢弃，训练全程不参与任何 loss —— 这是无监督域适应的定义。
4. **梯度自校准**（本 kernel 的关键）：CORAL 除以 4d² 后 loss 只有 ~5e-3，
   实测梯度范数 0.020 vs 病害 CE 5.32，相差 261 倍。μ=0.5 时 CORAL 只占总梯度
   的 0.19%，等于没开，跑完会得到一个假的「CORAL 无效」结论。因此开启
   --coral-autoscale，在首个 batch 实测梯度比并把 CORAL 放大到指定占比。

Kaggle 环境说明
--------------
- 预装 torch / torchvision，需补装 datasets（从 HuggingFace 拉 PlantVillage）
- 联网开启后可直接下载（约 813MB）
- 产物写到 /kaggle/working/runs，用 `kaggle kernels output` 拉回本地

用法（本地驱动）
--------------
    python -m kaggle kernels push -p kaggle\\coral_mtl --accelerator NvidiaTeslaT4
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"
SCRIPTS = {
    "train_coral_mtl.py": f"{RAW_BASE}/ml/src/train_coral_mtl.py",
    # eval_crossdomain.py 依赖同目录的 data.py / models.py，必须一起拉
    "eval_crossdomain.py": f"{RAW_BASE}/ml/src/eval_crossdomain.py",
    "data.py": f"{RAW_BASE}/ml/src/data.py",
    "models.py": f"{RAW_BASE}/ml/src/models.py",
}

WORK = "/kaggle/working"
SRC = f"{WORK}/src"
DATA_ROOT = f"{WORK}/data/plantvillage"
PLANTDOC_ROOT = f"{WORK}/data/plantdoc"
OUT_DIR = f"{WORK}/runs"
ZIP_PATH = f"{WORK}/PlantDoc.zip"

# PlantDoc 在 HuggingFace 上有公开 zip（与 kaggle/plantdoc_ft 同一来源）
MIRRORS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]


def log(msg: str) -> None:
    print(f"[kaggle] {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: str, attempts: int = 3) -> bool:
    """urllib 下载 + Range 断点续传。HF 对大文件支持 Range，断了不会白下。"""
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
                            log(f"  {pct} {human(done)} / "
                                f"{human(total) if total else '?'}  {human(speed)}/s")
                            last = now
            size = os.path.getsize(dest)
            if total and size < total:
                raise IOError(f"未下完：{human(size)} / {human(total)}")
            log(f"下载完成：{human(size)}")
            return True
        except Exception as exc:  # noqa: BLE001
            log(f"下载失败（第 {attempt} 次）：{type(exc).__name__}: {exc}")
            time.sleep(5)
    return False


def fetch_plantdoc() -> bool:
    """下载并解压 PlantDoc 到 DATA_ROOT（zip 内部顶层就是 train/ 与 test/）。"""
    if os.path.isdir(f"{PLANTDOC_ROOT}/train") and os.path.isdir(f"{PLANTDOC_ROOT}/test"):
        log("PlantDoc 已解压，跳过")
        return True

    for url in MIRRORS:
        if download(url, ZIP_PATH):
            break
    else:
        log("所有镜像都失败")
        return False

    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            bad = z.testzip()
            if bad:
                raise IOError(f"zip 损坏了：{bad}")
            z.extractall(PLANTDOC_ROOT)
    except Exception as exc:  # noqa: BLE001
        log(f"解压失败：{type(exc).__name__}: {exc}")
        return False

    ok = os.path.isdir(f"{PLANTDOC_ROOT}/train") and os.path.isdir(f"{PLANTDOC_ROOT}/test")
    if not ok:
        log(f"解压后目录不对：{sorted(os.listdir(PLANTDOC_ROOT))[:10]}")
        return False
    n_tr = sum(len(files) for _, _, files in os.walk(f"{PLANTDOC_ROOT}/train"))
    n_te = sum(len(files) for _, _, files in os.walk(f"{PLANTDOC_ROOT}/test"))
    log(f"图片数：train={n_tr} test={n_te}（期望 2670 / 252）")
    return True


def fetch_scripts() -> None:
    """从 GitHub 拉源码，保持单一真源（不在 Kaggle 维护副本）。"""
    os.makedirs(SRC, exist_ok=True)
    for name, url in SCRIPTS.items():
        path = os.path.join(SRC, name)
        log(f"拉取 {name}")
        urllib.request.urlretrieve(url, path)
        log(f"  {os.path.getsize(path)} 字节")



def run_train(script: str, mu: float, run_name: str) -> int:
    """跑一个 μ 配置。"""
    cmd = [
        sys.executable, "-u", script,
        "--model", "resnet18",
        "--epochs", "15",
        "--img-size", "224",
        "--batch-size", "64",
        "--lr", "0.0003",
        "--crop-weight", "0.4",
        "--coral-weight", str(mu),
        "--coral-warmup-epochs", "2",
        "--coral-autoscale",
        "--coral-grad-ratio", "0.05",
        "--download-src",
        "--src-root", DATA_ROOT,
        "--tgt-root", f"{PLANTDOC_ROOT}/train",
        "--out-dir", OUT_DIR,
        "--run-name", run_name,
        "--no-eval-tgt",  # 跨域评估统一在最后用 eval_crossdomain.py 做，口径一致
    ]
    log("执行: " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=WORK)
    log(f"训练退出码 {rc}")
    return rc


def run_eval(script: str, run_name: str) -> dict | None:
    """用仓库统一脚本做跨域评估，保证与基线同口径。"""
    run_dir = f"{OUT_DIR}/{run_name}"
    if not os.path.isdir(run_dir):
        log(f"警告：run 目录不存在 {run_dir}")
        return None
    cmd = [
        sys.executable, "-u", script,
        "--run-dir", run_dir,
        "--plantdoc-root", PLANTDOC_ROOT,
        "--split", "all",
        "--batch-size", "32",
        "--num-workers", "2",
    ]
    log("评估: " + " ".join(cmd))
    subprocess.call(cmd, cwd=SRC)  # eval 从 SRC 跑，能 import 同目录 data.py / models.py
    p = os.path.join(run_dir, "crossdomain.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> int:
    log("=" * 64)
    log(f"python {sys.version.split()[0]}")
    log(f"cwd={os.getcwd()}")

    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"gpu={torch.cuda.get_device_name(0)} vram={props.total_memory / 1e9:.1f}GB")
    log(f"nproc={os.cpu_count()}")

    log("安装 datasets（PlantVillage 下载依赖）")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

    fetch_scripts()
    train_script = f"{SRC}/train_coral_mtl.py"
    eval_script = f"{SRC}/eval_crossdomain.py"

    # 目标域：PlantDoc 从 HuggingFace 拉（与 kaggle/plantdoc_ft 同一来源）
    if not fetch_plantdoc():
        log("[错误] PlantDoc 准备失败，无法做无监督域适应")
        return 2

    # 执行顺序按「最便宜、最可能有效」排：
    #   1. no_coral（只多任务）—— 闭式实验已说明 CORAL 方向不对，这一档是最可能有效的
    #   2. mu=0.5 完整配置 —— 验证 CORAL 是否反而有害
    # 不做 mu=0.1/1.0 扫描：闭式 CORAL 重着色是 −1.35pp，调参救不回方向性错误。
    # T4 上每个配置约 25-30 分钟。
    CONFIGS = [
        ("coral_mtl_ablation_nocoral", 0.0),
        ("coral_mtl_resnet18_224_15ep_mu0.5", 0.5),
    ]
    results: dict[str, dict] = {}
    for run_name, mu in CONFIGS:
        log("-" * 64)
        log(f"开始 run={run_name} μ={mu}")
        rc = run_train(train_script, mu, run_name)
        if rc != 0:
            log(f"{run_name} 训练失败，跳过")
            continue
        cd = run_eval(eval_script, run_name)
        if cd:
            results[run_name] = {
                "mu": mu,
                "top1_acc": cd.get("top1_acc"),
                "top3_acc": cd.get("top3_acc"),
                "crop_acc": cd.get("crop_acc"),
                "n": cd.get("n"),
            }
            log(f"{run_name} -> top1={cd.get('top1_acc', 0):.4f} "
                f"crop={cd.get('crop_acc', 0):.4f}")

    # 消融：关掉 CORAL（只多任务）与关掉作物头（只 CORAL），定位收益来源
    log("-" * 64)
    log("消融：no_coral（只多任务）")
    rc = run_train(train_script, 0.0, "coral_mtl_ablation_nocoral")
    if rc == 0:
        cd = run_eval(eval_script, "coral_mtl_ablation_nocoral")
        if cd:
            results["coral_mtl_ablation_nocoral"] = {
                "mu": 0.0, "top1_acc": cd.get("top1_acc"),
                "top3_acc": cd.get("top3_acc"), "crop_acc": cd.get("crop_acc"),
                "n": cd.get("n"),
            }

    log("=" * 64)
    log("汇总（基线 ResNet18@224 15ep: top1=0.2818 crop=0.5113）")
    for name, r in results.items():
        log(f"  {name}: mu={r['mu']} top1={r['top1_acc']:.4f} "
            f"top3={r['top3_acc']:.4f} crop={r['crop_acc']:.4f} n={r['n']}")

    # 直接回答「CORAL 有没有用」：完整配置 vs 只多任务消融
    full = results.get("coral_mtl_resnet18_224_15ep_mu0.5")
    abl = results.get("coral_mtl_ablation_nocoral")
    verdict = None
    if full and abl:
        delta = (full["top1_acc"] - abl["top1_acc"]) * 100
        verdict = (
            f"CORAL 净效应 = {delta:+.2f}pp（完整 μ=0.5 {full['top1_acc']:.4f} "
            f"vs 只多任务 {abl['top1_acc']:.4f}）"
        )
        log(verdict)
        log("  > 0 说明 CORAL 有独立贡献；<= 0 说明收益全来自作物头，CORAL 是负担。")

    with open(f"{WORK}/coral_mtl_results.json", "w", encoding="utf-8") as f:
        json.dump({"runs": results, "coral_net_effect": verdict},
                  f, ensure_ascii=False, indent=2)
    log(f"结果已写 {WORK}/coral_mtl_results.json")

    if results:
        best = max(results.values(), key=lambda r: r["top1_acc"] or 0)
        log(f"最佳 top1={best['top1_acc']:.4f}（基线 0.2818，"
            f"变化 {(best['top1_acc'] - 0.2818) * 100:+.2f}pp）")

    log("产物清单:")
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in sorted(files):
            p = os.path.join(root, name)
            if os.path.getsize(p) > 1024:
                log(f"  {os.path.relpath(p, OUT_DIR)}  {os.path.getsize(p) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
