"""命令行推理：给一张叶片照片，输出 Top-3 病害预测。

这是 X-01 任务。作用有三：
    1. 人工抽查模型，看它到底在看什么（配合 Grad-CAM）
    2. 交付物之一——不用启动服务端也能验证模型
    3. 服务端 `server/inference.py` 的 Torch 后端逻辑与此一致，可互相印证

用法
----
    # 单张图
    python src/predict.py --image path/to/leaf.jpg

    # 指定模型
    python src/predict.py --image leaf.jpg --run-dir runs/mobilenet_v3_large_224_15ep

    # 整个目录，输出汇总表
    python src/predict.py --dir path/to/folder --out result.csv

    # 走 ONNX（需要先跑 export_onnx.py）
    python src/predict.py --image leaf.jpg --onnx

输出
----
    Top-3 的中文病名、英文类名、置信度，以及是否触发低置信度警告。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_transforms  # noqa: E402
from models import build_model  # noqa: E402
from class_names_cn import CLASS_CN  # noqa: E402

DEFAULT_RUN = ROOT / "runs" / "resnet18_224_15ep"

# 低于这个置信度就提示"不可靠"，与 SRS 里 US-05 的阈值一致
LOW_CONF_THRESHOLD = 0.60


def load_classes(run_dir: Path) -> list[str]:
    """读类别文件夹名列表，顺序与模型输出索引一致。"""
    cj = run_dir / "classes.json"
    if not cj.exists():
        raise FileNotFoundError(f"缺少 {cj}，无法确定类别顺序")
    data = json.loads(cj.read_text(encoding="utf-8"))
    # 兼容两种格式：['00_Apple_...'] 或 [{'index':0,'folder':'...'}]
    if data and isinstance(data[0], dict):
        return [d["folder"] for d in data]
    return list(data)


def pretty(folder: str) -> str:
    """'29_Tomato_Early_blight' → '番茄早疫病'（中文名缺失时退回可读格式）"""
    entry = CLASS_CN.get(folder)
    if entry:
        return entry[0]
    body = folder.split("_", 1)[1] if folder[:2].isdigit() else folder
    parts = body.split("_", 1)
    return f"{parts[0]} · {parts[1].replace('_', ' ')}" if len(parts) == 2 else body


class Predictor:
    """封装模型加载与单图推理。"""

    def __init__(self, run_dir: Path, img_size: int = 224, use_onnx: bool = False) -> None:
        self.run_dir = run_dir
        self.img_size = img_size
        self.classes = load_classes(run_dir)
        self.transform = build_transforms(img_size, train=False)
        self.use_onnx = use_onnx

        if use_onnx:
            import onnxruntime as ort

            onnx_path = run_dir / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"找不到 {onnx_path}，先运行：python src/export_onnx.py --run-dir {run_dir}"
                )
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
        else:
            ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
            model, _meta = build_model(model_name, len(self.classes), pretrained=False)
            model.load_state_dict(ck["model"])
            model.eval()
            self.model = model
            self.model_name = model_name
            self.ckpt_epoch = ck.get("epoch")

    def predict(self, img_path: Path, topk: int = 3) -> list[tuple[str, float]]:
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        if self.use_onnx:
            import numpy as np

            logits = self.sess.run(None, {self.input_name: x.numpy()})[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=1)
        else:
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)

        vals, idxs = probs.topk(min(topk, len(self.classes)), dim=1)
        return [
            (self.classes[i], float(v))
            for i, v in zip(idxs[0].tolist(), vals[0].tolist())
        ]


def print_result(img_path: Path, preds: list[tuple[str, float]]) -> None:
    top1_cls, top1_p = preds[0]
    print()
    print(f"图片    : {img_path.name}")
    print(f"Top-1   : {pretty(top1_cls):<42} {top1_p * 100:5.1f}%")
    for cls, p in preds[1:]:
        print(f"  Top-{preds.index((cls, p)) + 1}  : {pretty(cls):<42} {p * 100:5.1f}%")

    if top1_p < LOW_CONF_THRESHOLD:
        print(f"⚠ 置信度 {top1_p * 100:.1f}% 低于 {LOW_CONF_THRESHOLD * 100:.0f}% 阈值，结果不可靠")
        print("  真实拍摄场景下模型准确率约 28%（见 ml/artifacts/crossdomain.md），请人工核实")


def main() -> int:
    p = argparse.ArgumentParser(description="植物叶片病害命令行推理")
    p.add_argument("--image", default="", help="单张图片路径")
    p.add_argument("--dir", default="", help="图片目录（批量）")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN), help="模型产物目录")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--onnx", action="store_true", help="用 ONNX 推理而非 PyTorch")
    p.add_argument("--out", default="", help="批量模式的 CSV 输出路径")
    p.add_argument("--ext", default=".jpg,.jpeg,.png", help="批量模式的文件扩展名")
    args = p.parse_args()

    if not args.image and not args.dir:
        p.error("必须指定 --image 或 --dir")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()

    try:
        predictor = Predictor(run_dir, args.img_size, args.onnx)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return 1

    backend = "ONNX" if args.onnx else f"PyTorch ({predictor.model_name})"
    print(f"模型    : {run_dir.name}  [{backend}]")
    print(f"类别数  : {len(predictor.classes)}")

    # ---- 单张 ----
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[错误] 图片不存在: {img_path}")
            return 1
        preds = predictor.predict(img_path, args.topk)
        print_result(img_path, preds)
        return 0

    # ---- 批量 ----
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"[错误] 目录不存在: {dir_path}")
        return 1

    exts = tuple(e.strip().lower() for e in args.ext.split(","))
    files = sorted(
        f for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    )
    if not files:
        print(f"[错误] 目录里没有 {args.ext} 文件")
        return 1

    print(f"待处理  : {len(files)} 张")
    rows = []
    low_conf = 0
    for i, f in enumerate(files, 1):
        preds = predictor.predict(f, args.topk)
        top1_cls, top1_p = preds[0]
        if top1_p < LOW_CONF_THRESHOLD:
            low_conf += 1
        rows.append({
            "file": f.as_posix(),
            "top1": pretty(top1_cls),
            "top1_conf": f"{top1_p:.4f}",
            "top2": pretty(preds[1][0]) if len(preds) > 1 else "",
            "top2_conf": f"{preds[1][1]:.4f}" if len(preds) > 1 else "",
            "top3": pretty(preds[2][0]) if len(preds) > 2 else "",
            "top3_conf": f"{preds[2][1]:.4f}" if len(preds) > 2 else "",
            "low_conf": int(top1_p < LOW_CONF_THRESHOLD),
        })
        if i % 20 == 0 or i == len(files):
            print(f"  已处理 {i}/{len(files)}", flush=True)

    out_path = Path(args.out) if args.out else dir_path / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"已保存  : {out_path}")
    print(f"低置信度: {low_conf} / {len(files)} ({low_conf / len(files) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())