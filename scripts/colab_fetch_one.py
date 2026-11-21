"""调用 colab run 训练一个模型，并把 stdout 里的 base64 产物解包到本地。

配套 scripts/colab_train_one.py（在远端跑）。本脚本负责：
    1. 起 colab run（一次性 VM，跑完自动释放）
    2. 实时透传远端日志到终端
    3. 抓取 ===ARTIFACT_BEGIN=== / ===ARTIFACT_END=== 之间的 base64
    4. 解包到 ml/runs/<run_name>/

为什么不用 colab exec + 会话：
    免费版会话会被回收（本会话已被回收两次），/content 全丢。
    colab run 的 stdout 流回本地，训练一结束产物就落盘，不依赖会话存活。

用法：
    python scripts/colab_fetch_one.py swin_t --gpu T4
    python scripts/colab_fetch_one.py mobilenet_v3_large --epochs 15
    python scripts/colab_fetch_one.py resnet18 --skip-data      # VM 内已有数据
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_SCRIPT = ROOT / "scripts" / "colab_train_one.py"
RUNS_DIR = ROOT / "ml" / "runs"


def unpack(b64: str, dest_root: Path) -> Path | None:
    """把 base64 tarball 解到 dest_root/<run_name>。拒绝越界路径。"""
    raw = base64.b64decode(b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        members = []
        for m in tf.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                print(f"[skip] 可疑路径 {m.name}", file=sys.stderr)
                continue
            members.append(m)
        if not members:
            return None
        # 所有成员应共享同一个顶层目录名（run_name）
        top = Path(members[0].name).parts[0] if len(Path(members[0].name).parts) > 1 else ""
        dest = dest_root / top if top else dest_root
        dest.mkdir(parents=True, exist_ok=True)
        tf.extractall(dest_root, members=members, filter="data")
        return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="colab run 训练并取回产物")
    ap.add_argument("model", help="模型名")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=0, help="0 表示用远端默认")
    ap.add_argument("--lr", default="")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--timeout", type=float, default=0, help="0 表示不限")
    args = ap.parse_args()

    cmd = [
        "colab", "run",
        "--gpu", args.gpu,
        "--timeout", str(args.timeout),
        str(REMOTE_SCRIPT),
        args.model,
        "--epochs", str(args.epochs),
        "--img-size", str(args.img_size),
    ]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.lr:
        cmd += ["--lr", args.lr]
    if args.run_name:
        cmd += ["--run-name", args.run_name]
    if args.skip_data:
        cmd += ["--skip-data"]

    print("[local] " + " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    in_artifact = False
    b64_parts: list[str] = []
    meta: dict[str, str] = {}

    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip("\n")
        if s.strip() == "===ARTIFACT_BEGIN===":
            in_artifact = True
            continue
        if s.strip() == "===ARTIFACT_END===":
            in_artifact = False
            continue
        if in_artifact:
            if "=" in s and s.split("=", 1)[0] in {
                "RUN_NAME", "FILE_COUNT", "HAS_SUMMARY", "RC",
            }:
                k, v = s.split("=", 1)
                meta[k] = v
            else:
                b64_parts.append(s.strip())
            continue
        print(s, flush=True)

    rc = proc.wait()
    print(f"\n[local] colab run 退出码 {rc}")
    print(f"[local] meta {meta}")

    if not b64_parts:
        print("[local] 没有抓到产物。可能训练失败或 VM 超时。", file=sys.stderr)
        return 1

    b64 = "".join(b64_parts)
    print(f"[local] 解包 base64（{len(b64)} 字符）…")
    try:
        dest = unpack(b64, RUNS_DIR)
    except Exception as e:
        print(f"[local] 解包失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if dest is None:
        print("[local] 解包结果为空", file=sys.stderr)
        return 1

    print(f"[local] 已落盘 {dest}")
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(RUNS_DIR)}  {f.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
