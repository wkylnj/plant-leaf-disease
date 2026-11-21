"""把已训练好的 checkpoint 导出成 ONNX，并做数值一致性校验。

解决的问题
----------
训练脚本 train_colab.py 里有 ONNX 导出逻辑，但 Colab 环境缺 `onnxscript`
（torch>=2.5 的 dynamo 导出器必需），导致导出被 except 吞掉、只能看到一行
"ONNX 导出失败（不影响训练结果）"。本脚本可以在任意机器上对已有 best.pt
补做导出，不需要重训。

用法
----
    python src/export_onnx.py --run-dir runs/resnet18_224_15ep
    python src/export_onnx.py --run-dir runs/mobilenet_v3_large_224_15ep --opset 17

产物
----
    <run-dir>/model.onnx          导出的模型
    <run-dir>/onnx_meta.json      导出元信息（输入输出名、误差、大小）

校验
----
用 onnxruntime 跑同一批输入，与 PyTorch 输出比对最大绝对误差。
误差 < 1e-4 视为通过（float32 累加误差的正常量级）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model  # noqa: E402


def export(
    model: torch.nn.Module,
    img_size: int,
    onnx_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    """导出单个模型到 onnx_path。

    注意：torch >= 2.5 默认走 dynamo 导出器，需要 onnxscript。若想退回旧版
    TorchScript 导出器，传 dynamo=False。
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(model, dummy, str(onnx_path), **kwargs)


def verify(onnx_path: Path, model: torch.nn.Module, img_size: int) -> dict:
    """用 onnxruntime 与 PyTorch 对比输出，返回误差信息。"""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        return {"verified": False, "reason": f"缺少依赖: {e}"}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 固定随机种子，保证 PyTorch 与 ONNX 吃同一份输入
    torch.manual_seed(0)
    x = torch.randn(2, 3, img_size, img_size)

    with torch.no_grad():
        ref = model(x).cpu().numpy()
    got = sess.run(None, {input_name: x.numpy()})[0]

    max_diff = float(np.abs(ref - got).max())
    # 分类结果是否一致（更贴近实际用途）
    same_pred = bool((ref.argmax(1) == got.argmax(1)).all())

    return {
        "verified": True,
        "max_abs_diff": max_diff,
        "argmax_match": same_pred,
        "input_name": input_name,
        "output_names": [o.name for o in sess.get_outputs()],
        "batch_shape": list(sess.get_inputs()[0].shape),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="导出 checkpoint 为 ONNX 并校验")
    p.add_argument("--run-dir", required=True, help="训练产物目录，需含 best.pt")
    p.add_argument("--ckpt", default="best.pt", help="checkpoint 文件名")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", default="", help="输出路径，默认 <run-dir>/model.onnx")
    p.add_argument("--no-dynamic", action="store_true", help="不做 batch 维动态")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt_path}")
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    classes = ck["classes"]
    num_classes = len(classes)

    print(f"run_dir     : {run_dir}")
    print(f"model       : {model_name}")
    print(f"ckpt_epoch  : {ck.get('epoch', '?')}")
    print(f"ckpt_val_acc: {ck.get('val_acc', float('nan')):.4f}")
    print(f"num_classes : {num_classes}")

    # build_model 返回 (model, meta) 二元组
    model, _meta = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    onnx_path = Path(args.out).expanduser().resolve() if args.out else run_dir / "model.onnx"

    t0 = time.time()
    try:
        export(model, args.img_size, onnx_path, args.opset, not args.no_dynamic)
    except Exception as e:
        print(f"[失败] ONNX 导出异常: {type(e).__name__}: {e}")
        return 2
    elapsed = time.time() - t0

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"已导出      : {onnx_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    info = verify(onnx_path, model, args.img_size)
    if info.get("verified"):
        diff = info["max_abs_diff"]
        ok = diff < 1e-4 and info["argmax_match"]
        print(f"数值校验    : 最大绝对误差 {diff:.2e}, 分类一致 {info['argmax_match']} -> {'通过' if ok else '异常'}")
        info["passed"] = ok
    else:
        print(f"数值校验    : 跳过（{info.get('reason')}）")

    meta = {
        "model": model_name,
        "ckpt_epoch": ck.get("epoch"),
        "ckpt_val_acc": ck.get("val_acc"),
        "num_classes": num_classes,
        "img_size": args.img_size,
        "opset": args.opset,
        "size_mb": round(size_mb, 2),
        "export_seconds": round(elapsed, 2),
        "torch_version": torch.__version__,
        **info,
    }
    (run_dir / "onnx_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"元信息      : {run_dir / 'onnx_meta.json'}")

    return 0 if info.get("pass", info.get("passed", True)) else 3


if __name__ == "__main__":
    sys.exit(main())