"""推理层：ONNX Runtime 加载模型，或回退到 PyTorch / 假模型。

为什么要三种后端：
- `onnx`：生产路径。ONNX Runtime 不依赖 torch，服务器装包小、启动快、CPU 推理稳定。
- `torch`：训练完还没导出 ONNX 时的过渡方案，直接吃 best.pt。
- `fake`：**后端和安卓可以立刻开工**。A 的模型还没训完时，B/C 不用等，
  用假模型先把接口、联调、界面全跑通，之后只换一个文件就切到真模型。
  这是本周计划里"接口契约先冻结"的落地方式。

模型文件查找顺序：显式路径 → server/models/model.onnx → ml/runs/*/model.onnx（最新）。
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "server" / "models" / "model.onnx"
DEFAULT_CLASSES = REPO_ROOT / "server" / "models" / "classes.json"


# ---------------------------------------------------------------- 中文名
def load_class_cn() -> dict[str, str]:
    """加载 38 类中文映射。失败返回空字典（推理照常，只是标签显示英文）。"""
    try:
        import sys

        src = REPO_ROOT / "ml" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from class_names_cn import CLASS_CN  # type: ignore

        return {k: v[0] for k, v in CLASS_CN.items()}
    except Exception:
        return {}


def _short(label: str) -> str:
    """`00_Apple_Apple_scab` → `Apple_Apple_scab`，去掉两位索引前缀。"""
    parts = label.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else label


# ---------------------------------------------------------------- 预处理
def preprocess(img_bytes: bytes, img_size: int = 224) -> np.ndarray:
    """字节 → NCHW float32 张量。与训练时的验证集 transform 必须完全一致。

    训练用的是 Resize(int(size*256/224)) + CenterCrop(size)，这里照抄。
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = int(img_size * 256 / 224)
    # 短边缩放到 scale，再中心裁剪
    if w <= h:
        nw, nh = scale, max(scale, int(round(h * scale / w)))
    else:
        nh, nw = scale, max(scale, int(round(w * scale / h)))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - img_size) // 2, (nh - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    return arr[None, ...].astype(np.float32)  # 加 batch 维


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- 后端
class BasePredictor:
    name = "base"
    version = "0"
    classes: list[str] = []

    def predict(self, img_bytes: bytes) -> tuple[list[dict[str, Any]], float]:
        raise NotImplementedError


class FakePredictor(BasePredictor):
    """确定性假模型：把图片内容的哈希映射成概率分布。

    为什么不是纯随机：同一个 request 重放时结果应该一样，方便 C 调试界面。
    概率分布刻意做得"像真的"——top1 0.6~0.9，而不是均匀分布，
    这样 C 能真实测到进度条、置信度配色这些 UI 细节。
    """

    name = "fake"
    version = "0.0.0-fake"

    def __init__(self, classes: list[str]):
        self.classes = classes

    def predict(self, img_bytes: bytes) -> tuple[list[dict[str, Any]], float]:
        import hashlib

        t0 = time.perf_counter()
        h = hashlib.sha256(img_bytes).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        n = len(self.classes)
        logits = rng.normal(0, 1, size=n)
        # 人为抬高一个类，模拟"有明确判断"
        win = int.from_bytes(h[8:12], "big") % n
        logits[win] += 3.0 + (h[12] / 255.0) * 2.0
        probs = softmax(logits[None, :])[0]
        order = np.argsort(-probs)[:3]
        top3 = [
            {"label": self.classes[i], "label_cn": self._cn(self.classes[i]), "prob": round(float(probs[i]), 4)}
            for i in order
        ]
        return top3, (time.perf_counter() - t0) * 1000

    _cn_map: dict[str, str] = {}

    def _cn(self, label: str) -> str:
        if not self._cn_map:
            self._cn_map = load_class_cn()
        return self._cn_map.get(label, _short(label))


class OnnxPredictor(BasePredictor):
    name = "onnx"
    version = "1.0"

    def __init__(self, onnx_path: Path, classes: list[str], img_size: int = 224):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.classes = classes
        self.img_size = img_size
        # CPU 上 2 线程通常最划算；设 0 会让 ORT 用满所有核，反而和别的进程抢
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.version = f"1.0+{self.onnx_path.stat().st_mtime_ns}"
        self._cn_map = load_class_cn()

    def predict(self, img_bytes: bytes) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        x = preprocess(img_bytes, self.img_size)
        logits = self.sess.run([self.output_name], {self.input_name: x})[0]
        probs = softmax(np.asarray(logits, dtype=np.float32))[0]
        order = np.argsort(-probs)[:3]
        top3 = [
            {
                "label": self.classes[i],
                "label_cn": self._cn_map.get(self.classes[i], _short(self.classes[i])),
                "prob": round(float(probs[i]), 4),
            }
            for i in order
        ]
        return top3, (time.perf_counter() - t0) * 1000


class TorchPredictor(BasePredictor):
    """过渡方案：直接吃训练产出的 best.pt。仅在本机装有 torch 时可用。"""

    name = "torch"
    version = "1.0"

    def __init__(self, ckpt_path: Path, img_size: int = 224):
        import sys

        import torch

        self.torch = torch
        src = REPO_ROOT / "ml" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from models import build_model  # type: ignore

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        model_name = (ckpt.get("args") or {}).get("model", "resnet18")
        model, _meta = build_model(model_name, len(self.classes), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.img_size = img_size
        self.version = f"1.0+{Path(ckpt_path).stat().st_mtime_ns}"
        self._cn_map = load_class_cn()

    def predict(self, img_bytes: bytes) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        x = self.torch.from_numpy(preprocess(img_bytes, self.img_size))
        with self.torch.no_grad():
            logits = self.model(x[None, ...]).numpy()
        probs = softmax(logits)[0]
        order = np.argsort(-probs)[:3]
        top3 = [
            {
                "label": self.classes[i],
                "label_cn": self._cn_map.get(self.classes[i], _short(self.classes[i])),
                "prob": round(float(probs[i]), 4),
            }
            for i in order
        ]
        return top3, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- 工厂
def load_classes(path: Path | None = None) -> list[str]:
    """读类别清单。兼容两种格式：字符串列表，或 [{index,folder,name}]。"""
    p = Path(path) if path else DEFAULT_CLASSES
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw and isinstance(raw[0], dict):
        return [r.get("folder") or r.get("name") for r in raw]
    return list(raw)


def build_predictor(backend: str = "auto", model_dir: Path | None = None) -> BasePredictor:
    """按 backend 构造推理器。auto = onnx → torch → fake 依次降级。"""
    md = Path(model_dir) if model_dir else DEFAULT_ONNX.parent
    classes = load_classes(md / "classes.json") or load_classes(DEFAULT_CLASSES)

    if backend in ("auto", "onnx"):
        cand = [md / "model.onnx", DEFAULT_ONNX]
        # 再找训练目录里最新的
        runs = sorted(REPO_ROOT.glob("ml/runs/*/model.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        cand += runs
        for p in cand:
            if p.exists():
                try:
                    return OnnxPredictor(p, classes or load_classes(md / "classes.json"))
                except Exception as e:  # noqa: BLE001
                    print(f"[predictor] ONNX 加载失败 {p}: {e}")
        if backend == "onnx":
            raise RuntimeError("指定了 onnx 但没找到可用模型")

    if backend in ("auto", "torch"):
        cand = [md / "best.pt", DEFAULT_ONNX.parent / "best.pt"]
        cand += sorted(REPO_ROOT.glob("ml/runs/*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cand:
            if p.exists():
                try:
                    return TorchPredictor(p)
                except Exception as e:  # noqa: BLE001
                    print(f"[predictor] Torch 加载失败 {p}: {e}")
        if backend == "torch":
            raise RuntimeError("指定了 torch 但没找到可用权重")

    # 兜底：假模型。没有 classes.json 时也要能跑，用 38 个占位类名。
    if not classes:
        classes = [f"{i:02d}_class_{i}" for i in range(38)]
    print("[predictor] 未找到真实模型，使用 fake 后端（接口可联调，预测无意义）")
    return FakePredictor(classes)
