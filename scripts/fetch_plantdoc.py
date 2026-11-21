"""下载 PlantDoc 数据集（跨域测试用）。

为什么单独写脚本：
    curl 在本机走不通 huggingface.co（连接超时），但 Python 的 urllib 正常，
    可能与系统代理设置有关。用 Python 下载更可控，且能显示进度与断点续传。

用法：
    python scripts/fetch_plantdoc.py
    python scripts/fetch_plantdoc.py --mirror    # 走 hf-mirror.com

产物：
    ml/data/raw/plantdoc/PlantDoc.zip
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = ROOT / "ml" / "data" / "raw" / "plantdoc"
DEST = DEST_DIR / "PlantDoc.zip"

MIRRORS = {
    "official": "https://huggingface.co/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
    "mirror": "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip",
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"检测到已下载 {human(existing)}，尝试断点续传")

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        print(f"[失败] 无法打开 URL: {type(e).__name__}: {e}")
        return False

    total = resp.headers.get("Content-Length")
    total = int(total) + existing if total else 0
    mode = "ab" if existing and resp.status == 206 else "wb"
    if mode == "wb":
        existing = 0

    print(f"开始下载 → {dest}")
    if total:
        print(f"总大小 {human(total)}")

    t0 = time.time()
    done = existing
    last_report = 0.0
    try:
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_report >= 2.0:
                    speed = (done - existing) / max(now - t0, 1e-6)
                    pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                    print(f"  {pct}  {human(done)} / {human(total)}  {human(speed)}/s", flush=True)
                    last_report = now
    except Exception as e:
        print(f"[中断] {type(e).__name__}: {e}（已下载 {human(done)}，可重跑续传）")
        return False

    print(f"完成：{dest} ({human(dest.stat().st_size)})")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror", action="store_true", help="使用 hf-mirror.com 镜像")
    p.add_argument("--url", default="", help="自定义 URL")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    url = args.url or (MIRRORS["mirror"] if args.mirror else MIRRORS["official"])
    print(f"URL: {url}")

    ok = download(url, DEST, timeout=args.timeout)
    if not ok and not args.mirror:
        print("\n官方源失败，自动尝试镜像…")
        ok = download(MIRRORS["mirror"], DEST, timeout=args.timeout)

    if not ok:
        print("\n[结论] 下载失败。可选项：")
        print("  1. 换镜像：python scripts/fetch_plantdoc.py --mirror")
        print("  2. 浏览器手动下载后放到:", DEST)
        print("  3. 先跳过 E2，做其他实验")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())