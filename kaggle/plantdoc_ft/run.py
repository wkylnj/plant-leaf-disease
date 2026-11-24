"""Kaggle Kernel：PlantDoc 5 折有监督域适应微调（resnet18 + mobilenet_v3_large）

这个 kernel 负责把"本地只能冒烟验证"的实验放到 GPU 上跑完：

1. 下载 PlantDoc（HuggingFace `LamTNguyen/PlantDoc`）。
   注意：curl 在部分网络环境下连不上 huggingface.co，但 Python urllib 可以，
   所以这里照 scripts/fetch_plantdoc.py 的写法走 urllib + Range 断点续传，
   官方源失败自动切 hf-mirror.com。
2. 从 raw.githubusercontent 拉源码（保持单一真源，不在 Kaggle 维护副本）：
   finetune_plantdoc.py 及其依赖 eval_crossdomain.py / data.py / models.py。
3. 从挂载的 Kaggle Dataset 里找 PlantVillage 的 best.pt（checkpoint 约 60MB，
   不进 git，作为 private dataset `plantdoc-pv-ckpt` 挂载）。
4. 每个模型跑 5 折微调 + 同一套折的零样本 baseline，结果 json 打到 stdout。

用法（本地驱动）：
    python -m kaggle kernels push -p kaggle/plantdoc_ft --accelerator NvidiaTeslaT4
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
    "finetune_plantdoc.py": f"{RAW_BASE}/ml/src/finetune_plantdoc.py",
    "eval_crossdomain.py": f"{RAW_BASE}/ml/src/eval_crossdomain.py",
    "data.py": f"{RAW_BASE}/ml/src/data.py",
    "models.py": f"{RAW_BASE}/ml/src/models.py",
}

WORK = "/kaggle/working"
SRC = f"{WORK}/src"
DATA_ROOT = f"{WORK}/data/plantdoc"
OUT_DIR = f"{WORK}/out"
ZIP_PATH = f"{WORK}/PlantDoc.zip"

MIRRORS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]

MODELS = ["resnet18", "mobilenet_v3_large"]

# 微调超参：任务给定的配置（backbone 1e-4 / head 1e-3 / 4 轮 / ls 0.1 / bs 64）
EPOCHS = "4"
BATCH_SIZE = "64"
FOLDS = "5"


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest, attempts=3):
    """urllib 下载 + Range 续传。HF 对大文件支持 Range，断了不会白下。"""
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
                            log(f"  {pct} {human(done)} / {human(total) if total else '?'}  {human(speed)}/s")
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


def fetch_plantdoc():
    if os.path.isdir(f"{DATA_ROOT}/train") and os.path.isdir(f"{DATA_ROOT}/test"):
        log("PlantDoc 已解压，跳过")
        return True

    for url in MIRRORS:
        if download(url, ZIP_PATH):
            break
    else:
        log("所有镜像都失败")
        return False

    log("校验并解压 …")
    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            bad = z.testzip()
            if bad:
                raise IOError(f"zip 损坏于 {bad}")
            z.extractall(DATA_ROOT)
    except Exception as e:
        log(f"解压失败：{type(e).__name__}: {e}")
        return False

    # zip 内部顶层就是 train/ 与 test/，直接落在 DATA_ROOT 下
    ok = os.path.isdir(f"{DATA_ROOT}/train") and os.path.isdir(f"{DATA_ROOT}/test")
    if not ok:
        log(f"解压后目录不对：{sorted(os.listdir(DATA_ROOT))[:10]}")
        return False
    n_tr = sum(len(files) for _, _, files in os.walk(f"{DATA_ROOT}/train"))
    n_te = sum(len(files) for _, _, files in os.walk(f"{DATA_ROOT}/test"))
    log(f"图片数：train={n_tr} test={n_te}（期望 2670 / 252）")
    return True


def fetch_scripts():
    os.makedirs(SRC, exist_ok=True)
    for name, url in SCRIPTS.items():
        path = os.path.join(SRC, name)
        log(f"拉取 {name}")
        urllib.request.urlretrieve(url, path)
        log(f"  {os.path.getsize(path)} 字节")
    return True


def find_checkpoints():
    """在 /kaggle/input 下找所有 best.pt，按所在目录名归类。"""
    found = {}
    for root, _dirs, files in os.walk("/kaggle/input"):
        for name in files:
            if name.endswith(".pt"):
                found[os.path.basename(root)] = os.path.join(root, name)
                found.setdefault(name, os.path.join(root, name))
    return found


def main():
    log("=" * 70)
    log(f"python {sys.version.split()[0]}")
    log(f"cwd={os.getcwd()}")

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
    if not fetch_plantdoc():
        log("PlantDoc 不可用，终止")
        sys.exit(2)

    ckpts = find_checkpoints()
    log(f"发现的 checkpoint 目录：{sorted(set(os.path.dirname(p) for p in ckpts.values()))}")

    results = {}
    for model in MODELS:
        ckpt = None
        for key, path in ckpts.items():
            if model in key or model in os.path.dirname(path):
                ckpt = path
                break
        if ckpt is None:
            log(f"[{model}] 找不到 checkpoint，跳过")
            results[model] = {"error": "checkpoint not found"}
            continue

        out_json = f"{OUT_DIR}/{model}.json"
        cmd = [
            sys.executable, "-u", f"{SRC}/finetune_plantdoc.py",
            "--ckpt", ckpt,
            "--model", model,
            "--plantdoc-root", DATA_ROOT,
            "--out", out_json,
            "--folds", FOLDS,
            "--epochs", EPOCHS,
            "--batch-size", BATCH_SIZE,
            "--num-workers", "2",
        ]
        log(f"[{model}] 开始：{' '.join(cmd)}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        rc = subprocess.call(cmd, cwd=SRC, env=env)
        log(f"[{model}] 退出码 {rc}")

        if rc == 0 and os.path.exists(out_json):
            with open(out_json, encoding="utf-8") as f:
                r = json.load(f)
            results[model] = {
                "ckpt": os.path.basename(os.path.dirname(ckpt)),
                "zeroshot_top1": f"{r['zeroshot']['top1_mean']:.4f} ± {r['zeroshot']['top1_std']:.4f}",
                "finetuned_top1": f"{r['finetuned']['top1_mean']:.4f} ± {r['finetuned']['top1_std']:.4f}",
                "zeroshot_crop": f"{r['zeroshot']['crop_mean']:.4f} ± {r['zeroshot']['crop_std']:.4f}",
                "finetuned_crop": f"{r['finetuned']['crop_mean']:.4f} ± {r['finetuned']['crop_std']:.4f}",
                "delta_top1": r["paired_delta"]["top1_mean"],
                "delta_crop": r["paired_delta"]["crop_mean"],
                "folds_better": r["paired_delta"]["folds_better"],
                "n_folds": r["paired_delta"]["n_folds"],
                "overall_zeroshot": r["zeroshot"]["overall_top1_all_folds"],
                "raw": r,
            }
            log(f"[{model}] 零样本 {results[model]['zeroshot_top1']} → "
                f"微调 {results[model]['finetuned_top1']}")
        else:
            results[model] = {"error": f"rc={rc}"}

    # 把结果打到 stdout，本地可以直接从 kernel 日志里抓，不必先拉 output
    print("\n" + "=" * 70)
    print("ALL_RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print("ALL_RESULTS_JSON_END", flush=True)

    summary_path = f"{OUT_DIR}/all_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"已保存 {summary_path}")

    print("\n[结论摘要]")
    print(f"{'模型':<22}{'零样本 top-1':>20}{'微调 top-1':>20}{'配对差':>10}")
    for model, r in results.items():
        if "error" in r:
            print(f"{model:<22}{'失败: ' + r['error']:>20}")
            continue
        print(f"{model:<22}{r['zeroshot_top1']:>20}{r['finetuned_top1']:>20}"
              f"{r['delta_top1']:>+10.4f}")
    print("\n注意：这是「有监督域适应」设定（用了 PlantDoc 标签），"
          "与零样本跨域泛化的 28.18% 不是同一件事。")

    if all("error" in r for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
