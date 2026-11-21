#!/usr/bin/env python
"""把 Colab 上的训练产物拉回本地 runs/ 目录。

用法：
    python scripts/colab_fetch.py --session train
    python scripts/colab_fetch.py --session train --run-name resnet18_224_15ep
    python scripts/colab_fetch.py --session train --skip-weights   # 不要 .pt（省流量）

做三件事：
1. 在远端把 /content/runs/<run_name> 打包成 tar.gz（可选剔除 *.pt）
2. colab download 拉回本地临时目录
3. 解包到 D:\\...\\ml\\runs\\<run_name>\\

为什么不直接用 colab download 逐个文件：
    colab download 只接受单个文件路径，目录要先 tar。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = ROOT / "ml" / "runs"
REMOTE_RUNS = "/content/runs"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """跑一条命令，实时打印。"""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True, encoding="utf-8", errors="replace")


def list_remote_runs(session: str) -> list[str]:
    """列出远端有哪些 run 目录。"""
    p = run(["colab", "ls", "-s", session, REMOTE_RUNS], check=False)
    if p.returncode != 0:
        return []
    names = []
    for line in (p.stdout or "").splitlines():
        s = line.strip().rstrip("/")
        if s and not s.startswith("/"):
            names.append(s)
    return names


def pack_remote(session: str, run_name: str, skip_weights: bool) -> str:
    """远端打包，返回远端 tar 路径。"""
    tar_remote = f"/content/{run_name}.tar.gz"
    # --exclude 用通配，跳过权重能省几百 MB
    excludes = " --exclude='*.pt'" if skip_weights else ""
    cmd = (
        f"cd {REMOTE_RUNS} && tar -czf {tar_remote}{excludes} {run_name} "
        f"&& ls -lh {tar_remote}"
    )
    run(["colab", "exec", "-s", session, cmd])
    return tar_remote


def download(session: str, tar_remote: str, tar_local: Path) -> None:
    """把远端 tar 拉到本地。"""
    tar_local.parent.mkdir(parents=True, exist_ok=True)
    if tar_local.exists():
        tar_local.unlink()
    run(["colab", "download", "-s", session, tar_remote, str(tar_local)])


def unpack(tar_local: Path, dest_root: Path) -> Path:
    """解包到 dest_root，返回解出来的 run 目录。"""
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_local, "r:gz") as tf:
        members = tf.getmembers()
        # 安全检查：不允许绝对路径或 ..
        for m in members:
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise RuntimeError(f"tar 里有可疑路径：{m.name}")
        tf.extractall(dest_root)
    tops = {Path(m.name).parts[0] for m in members if m.name.strip()}
    if len(tops) != 1:
        raise RuntimeError(f"tar 顶层目录不唯一：{tops}")
    return dest_root / tops.pop()


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取 Colab 训练产物")
    ap.add_argument("--session", default="train", help="Colab 会话名")
    ap.add_argument("--run-name", default=None, help="只拉这一轮；不填则拉全部")
    ap.add_argument("--skip-weights", action="store_true", help="不下载 .pt 权重")
    ap.add_argument("--keep-tar", action="store_true", help="保留本地 tar 包")
    args = ap.parse_args()

    if shutil.which("colab") is None:
        print("找不到 colab 命令，先 pip install google-colab-cli", file=sys.stderr)
        return 1

    names = [args.run_name] if args.run_name else list_remote_runs(args.session)
    if not names:
        print(f"远端 {REMOTE_RUNS} 下没有 run 目录（会话 {args.session} 对吗？）", file=sys.stderr)
        return 1

    print(f"待拉取：{names}")
    for name in names:
        print(f"\n=== {name} ===")
        tar_remote = pack_remote(args.session, name, args.skip_weights)
        with tempfile.TemporaryDirectory() as td:
            tar_local = Path(td) / f"{name}.tar.gz"
            download(args.session, tar_remote, tar_local)
            if not tar_local.exists() or tar_local.stat().st_size == 0:
                print(f"下载失败：{tar_local} 不存在或为空", file=sys.stderr)
                continue
            size_mb = tar_local.stat().st_size / 1e6
            print(f"下载完成 {size_mb:.1f} MB，开始解包…")
            out = unpack(tar_local, LOCAL_RUNS)
            print(f"解包到 {out}")
            if args.keep_tar:
                keep = LOCAL_RUNS / tar_local.name
                shutil.copy2(tar_local, keep)
                print(f"tar 已保留 {keep}")

    print("\n本地 runs/ 现状：")
    if LOCAL_RUNS.exists():
        for d in sorted(LOCAL_RUNS.iterdir()):
            if d.is_dir():
                files = sorted(p.name for p in d.iterdir() if p.is_file())
                print(f"  {d.name}/  {len(files)} 个文件: {', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
