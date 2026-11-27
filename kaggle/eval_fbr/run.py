"""Kaggle Kernel：评估 FBR 训练出的权重在 PlantDoc 上的跨域准确率。

为什么单独建一个 kernel
----------------------
`plant-leaf-fbr` 的输出目录里混着整个合成数据集（1.6 GB），
`kaggle kernels output` 只能整包下载，本地断了 3 次都拿不回来。

Kaggle 支持把另一个 kernel 的输出挂载为**输入**（`kernel_sources`），
所以这里直接把 `plant-leaf-fbr` 的输出挂到 `/kaggle/input/`，
权重就地读取，不经过本地磁盘。评估只需要几百 KB 的 json，拉回来很快。

评估口径与 `ml/src/eval_crossdomain.py` 完全一致：
    - PlantDoc 全量（27 类映射到我们 38 类）
    - top-1 / top-3 / 作物级 / 平均置信度
唯一区别：数据用 PlantDoc 官方 test split（252 张），
因为完整 2920 张的评估要 20+ 分钟，而 test split 足够判断路线成败。
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

RAW_BASE = "https://raw.githubusercontent.com/wkylnj/plant-leaf-disease/main"

SCRIPTS = {
    "eval_crossdomain.py": f"{RAW_BASE}/ml/src/eval_crossdomain.py",
    "data.py": f"{RAW_BASE}/ml/src/data.py",
    "models.py": f"{RAW_BASE}/ml/src/models.py",
}

PLANTDOC_URLS = [
    "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
]

WORK = "/kaggle/working"
SRC = os.path.join(WORK, "src")
DATA = os.path.join(WORK, "data")


def log(msg):
    print(f"[eval-fbr] {msg}", flush=True)


def fetch_scripts():
    os.makedirs(SRC, exist_ok=True)
    for name, url in SCRIPTS.items():
        dst = os.path.join(SRC, name)
        log(f"下载 {name}")
        urllib.request.urlretrieve(url, dst)
    log(f"脚本就绪：{len(SCRIPTS)} 个")


def dump_input_tree(max_entries=120):
    """打印 /kaggle/input 的目录结构，用来确认 kernel_sources 是否挂上。"""
    log("--- /kaggle/input 结构 ---")
    count = 0
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        log(f"  {'  ' * depth}{root}/  ({len(files)} files)")
        for f in files[:8]:
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p) / 1e6
            except OSError:
                sz = -1
            log(f"  {'  ' * depth}  - {f}  {sz:.1f} MB")
            count += 1
            if count >= max_entries:
                log("  ...（截断）")
                return
    if count == 0:
        log("  （空）")


def extract_from_output_zip():
    """挂载进来的是整个 `_output_.zip`（5.3 GB），在里面找 FBR 的权重。

    Kaggle 的 kernel_sources 挂的是**输出原样**——FBR 那次输出目录里
    混着整个合成数据集，被打成一个 5.3 GB 的 zip。好消息是 zip 就在
    本地磁盘上，`zipfile` 只需读末尾的中央目录，再 seek 到目标条目，
    不用解压 6600 张图。
    """
    import zipfile

    zips = sorted(glob.glob("/kaggle/input/**/_output_.zip", recursive=True))
    if not zips:
        log("没有找到 _output_.zip")
        return None

    zpath = zips[0]
    log(f"打开 {zpath}  {os.path.getsize(zpath) / 1e6:.1f} MB")
    want = re.compile(r"runs/.*fbr.*/(best\.pt|classes\.json|config\.json|summary\.json|history\.csv)$")

    dest_root = os.path.join(WORK, "runs")
    extracted = []
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        log(f"zip 内 {len(names)} 个条目")
        hits = [n for n in names if want.search(n)]
        log(f"命中 {len(hits)} 个")
        for name in hits:
            info = zf.getinfo(name)
            # 只保留 runs/ 之后的部分，去掉 data/ 等前缀
            rel = name[name.index("runs/"):]
            dest = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            log(f"  解压 {name}  {info.file_size / 1e6:.2f} MB")
            with zf.open(info) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted.append(dest)

    for p in extracted:
        if p.endswith("best.pt"):
            return p
    return None


def find_ckpt():
    """在挂载的 kernel 输出里找 FBR 的 best.pt。"""
    patterns = [
        "/kaggle/input/**/runs/*fbr*/best.pt",
        "/kaggle/input/**/*fbr*/best.pt",
        "/kaggle/input/**/best.pt",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            log(f"命中模式 {pat} → {len(hits)} 个")
            for h in hits:
                log(f"  {h}  {os.path.getsize(h) / 1e6:.1f} MB")
            return hits[0]
    # 没解压的目录，说明挂的是输出 zip
    return extract_from_output_zip()


def download_plantdoc():
    root = os.path.join(DATA, "raw", "plantdoc")
    marker = os.path.join(root, ".done")
    if os.path.exists(marker):
        log("PlantDoc 已存在")
        return root

    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(DATA, "PlantDoc.zip")
    os.makedirs(DATA, exist_ok=True)

    ok = False
    for url in PLANTDOC_URLS:
        try:
            log(f"下载 {url}")
            urllib.request.urlretrieve(url, zpath)
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            log(f"失败 {type(e).__name__}: {e}")

    if not ok:
        raise RuntimeError("PlantDoc 所有源都下载失败")

    log(f"解压 {os.path.getsize(zpath) / 1e6:.1f} MB")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(root)
    open(marker, "w").close()
    log("PlantDoc 就绪")
    return root


def main():
    log("=== 开始 ===")
    fetch_scripts()

    ckpt = find_ckpt()
    if ckpt is None:
        log("!!! 找不到 best.pt")
        dump_input_tree()
        return 2

    pd_root = download_plantdoc()

    # eval_crossdomain.py 会把 json 写进 --run-dir，而 /kaggle/input 是只读的，
    # 所以先把整个 run 目录（含 best.pt / classes.json / config.json）复制到
    # /kaggle/working 下再评估。
    src_run = os.path.dirname(ckpt)
    dst_run = os.path.join(WORK, "runs", os.path.basename(src_run))
    if not os.path.exists(os.path.join(dst_run, "best.pt")):
        log(f"复制 run 目录 {src_run} -> {dst_run}")
        shutil.copytree(src_run, dst_run, dirs_exist_ok=True)
    ckpt = os.path.join(dst_run, "best.pt")
    log(f"评估用权重 {ckpt}  {os.path.getsize(ckpt) / 1e6:.1f} MB")

    out_dir = os.path.join(WORK, "eval")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", os.path.join(SRC, "eval_crossdomain.py"),
        "--run-dir", dst_run,
        "--plantdoc-root", pd_root,
        "--split", "test",
        "--img-size", "224",
        "--batch-size", "32",
        "--num-workers", "2",
    ]
    log("执行: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)

    # eval_crossdomain 会把 json 写进 run 目录（可能在只读的 /kaggle/input 下）
    result = None
    for cand in glob.glob(os.path.join(os.path.dirname(ckpt), "crossdomain*.json")):
        try:
            result = json.load(open(cand, encoding="utf-8"))
            log(f"读取 {cand}")
            break
        except Exception:  # noqa: BLE001
            continue

    if result is None:
        log("!!! 没拿到 crossdomain.json，从 stdout 里找关键行")
        for line in proc.stdout.splitlines():
            if any(k in line for k in ("top1", "top-1", "作物", "crop", "acc")):
                log(f"  {line.strip()}")

    payload = {
        "ckpt": ckpt,
        "plantdoc_root": pd_root,
        "split": "test",
        "result": result,
        "returncode": proc.returncode,
    }
    out_path = os.path.join(out_dir, "fbr_crossdomain_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log(f"产物 {out_path}")
    log("=== 完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
