"""FastAPI 服务：/health /predict /feedback /stats

接口契约（docs/计划.md §2，B 与 C 的唯一交集，冻结后不许单方面改）：

    GET  /health
        → {"status":"ok","model":"onnx","version":"1.0+...","classes":38}

    POST /predict        multipart/form-data，字段名必须是 file
        → {"request_id":"a1b2...", "top3":[{"label","label_cn","prob"},...],
           "model":"onnx", "elapsed_ms":42.1}

    POST /feedback       application/json
        {"request_id":"a1b2...", "is_correct":true, "true_label":"00_Apple_Apple_scab"}
        → {"ok":true,"feedback_id":1}

    GET  /stats
        → {"predictions":n,"feedback":n,"feedback_correct":n,"feedback_accuracy":0.9,"users":n}

关键约束：`request_id` 与 `predictions.id` 是同一个值，这是 /feedback 唯一的关联线索。

启动：
    cd server && python -m uvicorn main:app --host 0.0.0.0 --port 8000
    # 假模型（无权重也能联调）
    PREDICT_BACKEND=fake python -m uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db import Store
from inference import build_predictor

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", REPO_ROOT / "server" / "uploads"))
DB_PATH = Path(os.environ.get("DB_PATH", REPO_ROOT / "server" / "app.db"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
SAVE_UPLOADS = os.environ.get("SAVE_UPLOADS", "1") == "1"

app = FastAPI(
    title="植物叶片病害识别 API",
    description="软件工程课程大作业 · Alpha 版本。上传叶片照片，返回 top-3 病害判断。",
    version="0.1.0",
)

# 安卓模拟器/真机跨域不敏感，但 /docs 页面调试需要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = Store(DB_PATH)
predictor = build_predictor(os.environ.get("PREDICT_BACKEND", "auto"))
if SAVE_UPLOADS:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackIn(BaseModel):
    request_id: str = Field(..., min_length=1, description="/predict 返回的 request_id")
    is_correct: bool
    true_label: str | None = Field(None, description="用户认为的正确类别（可选）")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": predictor.name,
        "version": predictor.version,
        "classes": len(predictor.classes),
        "uptime_s": round(time.time() - _STARTED, 1),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")
    # 只做最基本的魔数检查，不解析 EXIF——解析失败会让用户莫名其妙传不上去
    if not data[:4] in (b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\x89PNG", b"\xff\xd8\xff\xdb") and not data.startswith(b"\xff\xd8"):
        raise HTTPException(status_code=400, detail="只支持 JPEG/PNG")

    try:
        top3, elapsed_ms = predictor.predict(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    image_path = None
    if SAVE_UPLOADS:
        import uuid

        name = f"{uuid.uuid4().hex[:16]}.jpg"
        (UPLOAD_DIR / name).write_bytes(data)
        image_path = str(UPLOAD_DIR / name)

    request_id = store.add_prediction(
        top3=top3,
        model_name=predictor.name,
        elapsed_ms=elapsed_ms,
        image_path=image_path,
    )
    return {
        "request_id": request_id,
        "top3": top3,
        "model": predictor.name,
        "elapsed_ms": round(elapsed_ms, 1),
    }


@app.post("/feedback")
def feedback(body: FeedbackIn) -> dict[str, Any]:
    if store.get_prediction(body.request_id) is None:
        raise HTTPException(status_code=404, detail=f"request_id 不存在: {body.request_id}")
    fid = store.add_feedback(body.request_id, body.is_correct, body.true_label)
    return {"ok": True, "feedback_id": fid}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return store.stats()


@app.get("/recent")
def recent(limit: int = 20) -> list[dict[str, Any]]:
    return store.recent(min(max(limit, 1), 200))


@app.post("/users/{device_id}")
def touch_user(device_id: str) -> dict[str, str]:
    return {"user_id": store.touch_user(device_id)}


_STARTED = time.time()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
