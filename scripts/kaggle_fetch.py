"""从 Kaggle Kernel 拉回训练产物。

用法：
    python scripts/kaggle_fetch.py plant-leaf-swin-t --run-dir ml/runs/swin_t_224_15ep
    python scripts/kaggle_fetch.py plant-leaf-da-aug-mobilenet --run-dir ml/runs/mobilenet_v3_large_daaug_224_15ep
    python scripts/kaggle_fetch.py plant-leaf-swin-t --status      # 只看状态

说明：
- `kaggle kernels output` 只能下载 kernel 完成后的 /kaggle/working 内容
- 大文件（best.pt ~100MB）也会一起下载，Kaggle 单文件限制 20GB，够用
- 下载后把 runs/<run_name>/ 里的文件平铺到 --run-dir 下
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kaggle(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def status(slug: str) -> str:
    r = kaggle("kernels", "status", slug)
    out = (r.stdout or "").strip()
    print(out or (r.stderr or "").strip())
    return out


def fetch(slug: str, run_dir: Path, run_name: str | None, keep_tmp: bool = False) -> int:
    st = status(slug)
    if "COMPLETE" not in st:
        print(f"[!] kernel 尚未完成，当前状态: {st}")
        print("    稍后重试，或打开 https://www.kaggle.com/code/" + slug + " 查看进度")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="kaggle_out_"))
    print(f"[*] 下载产物到临时目录 {tmp}")
    r = kaggle("kernels", "output", slug, "-p", str(tmp))
    if r.returncode != 0:
        print("[!] 下载失败:", (r.stderr or r.stdout).strip()[:800])
        return 1

    # 产物结构：<tmp>/runs/<run_name>/*.pt 等
    candidates = list(tmp.rglob("best.pt"))
    if not candidates:
        print("[!] 未找到 best.pt，实际下载内容：")
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                print(f"    {p.relative_to(tmp)}  {p.stat().st_size/1e6:.2f} MB")
        # 至少把 summary/json 拉回来
        src = next((p for p in tmp.rglob("summary.json")), None)
        if src is None:
            return 1
        candidates = [src]

    src_dir = candidates[0].parent
    print(f"[*] 找到产物目录 {src_dir.relative_to(tmp)}，内容：")
    for p in sorted(src_dir.iterdir()):
        if p.is_file():
            print(f"    {p.name}  {p.stat().st_size/1e6:.2f} MB")

    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.is_file():
            shutil.copy2(p, run_dir / p.name)
            n += 1
    print(f"[OK] 复制 {n} 个文件到 {run_dir}")

    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"[i] 临时目录保留: {tmp}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="kernel slug，如 plant-leaf-swin-t（可省略用户名前缀）")
    ap.add_argument("--user", default="wangkaiyu66")
    ap.add_argument("--run-dir", default="", help="本地目标目录，默认 ml/runs/<kernel-slug>")
    ap.add_argument("--run-name", default="", help="kernel 内的 run 名，用于定位")
    ap.add_argument("--status", action="store_true", help="只查状态不下载")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    slug = args.slug if "/" in args.slug else f"{args.user}/{args.slug}"

    if args.status:
        status(slug)
        return 0

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir = ROOT / "ml" / "runs" / args.slug.split("/")[-1]

    return fetch(slug, run_dir, args.run_name or None, args.keep_tmp)


if __name__ == "__main__":
    sys.exit(main())
