"""环境自检：确认训练与可视化所需的依赖是否齐全。

用法:
    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = ["numpy", "pandas", "torch", "torchvision", "sklearn", "skimage", "cv2"]
OPTIONAL = ["matplotlib", "tqdm", "onnx", "onnxruntime", "seaborn"]


def probe(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - 只为自检，不需要细分异常
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", "?")
    return True, str(version)


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 60)

    missing_required = []
    for name in REQUIRED:
        ok, info = probe(name)
        print(f"{name:14s} {'OK  ' if ok else 'MISS'} {info}")
        if not ok:
            missing_required.append(name)

    print("-" * 60)
    for name in OPTIONAL:
        ok, info = probe(name)
        print(f"{name:14s} {'OK  ' if ok else 'MISS'} {info}  (可选)")

    data_root = Path(__file__).resolve().parents[1] / "ml" / "data" / "processed" / "plantvillage"
    print("-" * 60)
    if data_root.is_dir():
        for split in ("train", "val", "test"):
            split_dir = data_root / split
            if not split_dir.is_dir():
                print(f"{split:6s} 缺失")
                continue
            n_img = sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            n_cls = sum(1 for d in split_dir.iterdir() if d.is_dir())
            print(f"{split:6s} {n_cls:3d} 类  {n_img:6d} 张")
    else:
        print(f"数据目录不存在: {data_root}")

    if missing_required:
        print("-" * 60)
        print("缺少必需依赖: " + ", ".join(missing_required))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
