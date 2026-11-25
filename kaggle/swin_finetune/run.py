"""Kaggle Kernel：PlantDoc 5 折有监督域适应微调（Swin-T）

与 kaggle/plantdoc_ft/ 的差别只有两点：换模型、换 checkpoint 来源。

Swin-T 的 best.pt（105MB）**不在** 私有数据集 `wangkaiyu66/plantdoc-pv-ckpt` 里
（那个数据集只存了 resnet18 / mobilenet_v3_large）。所以本 kernel 按以下顺序找权重：

1. `/kaggle/input/**/best.pt` 里 `args.model == "swin_t"` 的那个（推荐做法：
   把 ml/runs/swin_t_224_15ep/best.pt 单独上传成一个 Kaggle Dataset，
   例如 `wangkaiyu66/plantdoc-swin-ckpt`，然后在 kernel 页面 Add Data）。
   注意不能塞进 plantdoc-pv-ckpt：那个数据集已挂载且被 plantdoc_ft kernel 复用，
   改内容会让旧 kernel 的输入版本号漂移。
2. 环境变量 `SWIN_CKPT_URL`（GitHub Release 直链）。设置了就下载到
   /kaggle/working/ckpt/swin_t/best.pt 再用。105MB 走 GitHub Release 可行，
   但不进 git 仓库本体——只作为 Release 附件。
3. 都没有 → 打印明确的操作提示后退出码 2，不静默跑错模型。

其余流程（下载 PlantDoc、拉源码、跑 5 折 + 同折零样本 baseline、结果打 stdout）
与 plantdoc_ft 完全一致，便于两个 kernel 的结果直接并列。

用法（本地驱动，需要用户手动执行，不要自动 push）：
    kaggle kernels push -p kaggle/swin_finetune
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
CKPT_DIR = f"{WORK}/ckpt/swin_t"

MIRRORS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]

MODEL = "swin_t"

# 微调超参：与 resnet18 那轮保持一致（backbone 1e-4 / head 1e-3 / 4 轮 / ls 0.1 / bs 64）。
# 唯一可以考虑调整的是 batch：Swin-T 显存约为 ResNet18 的 3 倍，T4 16GB 上
# bs=64 @224 通常还放得下；若 OOM 把下面改成 32，lr 不变（AdamW 对 bs 不敏感，
# 且这里只跑 4 轮，不值得为改 bs 再调 lr）。
EPOCHS = "4"
BATCH_SIZE = "64"
FOLDS = "5"

MIN_CKPT_BYTES = 50 * 1024 * 1024  # 105MB 的权重；小于这个值基本是下载被截断


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest, attempts=3):
    """urllib 下载 + Range 续传（HF/GitHub 都支持 Range）。"""
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


def _peek_model_name(path):
    """只读 checkpoint 的 args，判断是不是 swin_t 的权重。

    105MB 的权重逐个 load 一遍要几十秒，但 /kaggle/input 下可能同时挂着
    resnet18 / mobilenet / swin 三份 best.pt，不看内容没法区分——文件名都叫 best.pt。
    这里只读一次、只对候选文件读，可以接受。
    """
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"  读取失败 {path}：{type(e).__name__}: {e}")
        return None, None
    name = str(ck.get("args", {}).get("model", "")).lower()
    return name, ck.get("classes")


def find_swin_ckpt():
    """按「挂载数据集 → SWIN_CKPT_URL」的顺序找 Swin-T 的 best.pt。"""
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for name in files:
            if not name.endswith(".pt"):
                continue
            path = os.path.join(root, name)
            size = os.path.getsize(path)
            # 目录名里带 swin 的先试，避免每次都要 load 三个大文件
            prio = 0 if "swin" in path.lower() else 1
            candidates.append((prio, path, size))
    candidates.sort()

    for prio, path, size in candidates:
        if prio == 0 and "swin" in path.lower() and size > MIN_CKPT_BYTES:
            log(f"按目录名命中：{path}（{human(size)}）")
            model_name, _classes = _peek_model_name(path)
            if model_name == MODEL:
                return path
            log(f"  但 args.model={model_name!r}，继续找")

    for _prio, path, size in candidates:
        if size <= MIN_CKPT_BYTES:
            continue
        log(f"检查 {path}（{human(size)}）")
        model_name, _classes = _peek_model_name(path)
        if model_name == MODEL:
            log(f"  命中：args.model={MODEL}")
            return path

    # 兜底：GitHub Release 直链
    url = os.environ.get("SWIN_CKPT_URL", "").strip()
    if url:
        os.makedirs(CKPT_DIR, exist_ok=True)
        dest = os.path.join(CKPT_DIR, "best.pt")
        log(f"挂载数据集里没有 Swin-T 权重，改从 SWIN_CKPT_URL 下载：{url}")
        if download(url, dest) and os.path.getsize(dest) > MIN_CKPT_BYTES:
            model_name, _classes = _peek_model_name(dest)
            if model_name == MODEL:
                return dest
            log(f"  下载到的权重 args.model={model_name!r}，不是 {MODEL}")
        return None

    log("")
    log("!" * 70)
    log("找不到 Swin-T 的 best.pt。二选一：")
    log("  A) 把本地 ml/runs/swin_t_224_15ep/best.pt（约 105MB）上传成私有 Dataset")
    log("     （例如 wangkaiyu66/plantdoc-swin-ckpt），在本 kernel 的 Data 里挂载。")
    log("     不要塞进 wangkaiyu66/plantdoc-pv-ckpt——那个被 plantdoc_ft kernel 复用。")
    log("  B) 把该文件作为 GitHub Release 附件发布，然后设置环境变量")
    log("     SWIN_CKPT_URL=<release 直链> 后重跑本 kernel。")
    log("!" * 70)
    return None


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

    ckpt = find_swin_ckpt()
    if ckpt is None:
        sys.exit(2)

    out_json = f"{OUT_DIR}/{MODEL}.json"
    cmd = [
        sys.executable, "-u", f"{SRC}/finetune_plantdoc.py",
        "--ckpt", ckpt,
        "--model", MODEL,
        "--plantdoc-root", DATA_ROOT,
        "--out", out_json,
        "--folds", FOLDS,
        "--epochs", EPOCHS,
        "--batch-size", BATCH_SIZE,
        "--num-workers", "2",
    ]
    log(f"[{MODEL}] 开始：{' '.join(cmd)}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc = subprocess.call(cmd, cwd=SRC, env=env)
    log(f"[{MODEL}] 退出码 {rc}")

    results = {}
    if rc == 0 and os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            r = json.load(f)
        results[MODEL] = {
            "ckpt": ckpt,
            "ckpt_epoch": r.get("ckpt_epoch"),
            "ckpt_pv_val_acc": r.get("ckpt_pv_val_acc"),
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
        log(f"[{MODEL}] 零样本 {results[MODEL]['zeroshot_top1']} → "
            f"微调 {results[MODEL]['finetuned_top1']}")
    else:
        results[MODEL] = {"error": f"rc={rc}"}

    print("\n" + "=" * 70)
    print("ALL_RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print("ALL_RESULTS_JSON_END", flush=True)

    summary_path = f"{OUT_DIR}/all_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"已保存 {summary_path}")

    print("\n[结论摘要]")
    r = results[MODEL]
    if "error" in r:
        print(f"{MODEL:<22}{'失败: ' + r['error']:>20}")
        sys.exit(1)
    print(f"{'模型':<22}{'零样本 top-1':>20}{'微调 top-1':>20}{'配对差':>10}")
    print(f"{MODEL:<22}{r['zeroshot_top1']:>20}{r['finetuned_top1']:>20}"
          f"{r['delta_top1']:>+10.4f}")
    print("\n对照（同一套 5 折，来自本地已完成的实验）：")
    print("  resnet18  零样本 0.2818 → 微调 0.6973（配对差 +0.4155，5/5 折提升）")
    print("\n注意：这是「有监督域适应」设定（用了 PlantDoc 标签），"
          "与零样本跨域泛化的 34.93% 不是同一件事。")


if __name__ == "__main__":
    main()
