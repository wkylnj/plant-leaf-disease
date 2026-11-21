"""SQLite 存储层：预测记录与用户反馈。

设计要点（见 docs/计划.md §1）：
- `predictions.id` 与 API 返回的 `request_id` 是同一个值，这是 /feedback 关联的唯一线索。
- 用 WAL 模式，允许多线程并发读；FastAPI 默认在线程池里跑同步端点，需要 check_same_thread=False。
- 建表用 IF NOT EXISTS，服务启动时自动迁移，不需要单独的建库脚本。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    image_path  TEXT,
    top1_label  TEXT NOT NULL,
    top1_prob   REAL NOT NULL,
    top3_json   TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    elapsed_ms  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    is_correct    INTEGER NOT NULL,
    true_label    TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_pred ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);

CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    device_id  TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Store:
    """线程安全的极简 SQLite 封装。

    每个线程一个连接（sqlite3 连接不跨线程共享），用 threading.local 存。
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------ 连接
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------ 写
    def add_prediction(
        self,
        top3: list[dict[str, Any]],
        model_name: str,
        elapsed_ms: float,
        image_path: str | None = None,
        request_id: str | None = None,
    ) -> str:
        rid = request_id or uuid.uuid4().hex[:16]
        top1 = top3[0] if top3 else {"label": "", "prob": 0.0}
        self._conn().execute(
            "INSERT INTO predictions (id, created_at, image_path, top1_label, top1_prob, top3_json, model_name, elapsed_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                rid,
                _now(),
                image_path,
                top1.get("label", ""),
                float(top1.get("prob", 0.0)),
                json.dumps(top3, ensure_ascii=False),
                model_name,
                float(elapsed_ms),
            ),
        )
        self._conn().commit()
        return rid

    def add_feedback(self, prediction_id: str, is_correct: bool, true_label: str | None = None) -> int:
        cur = self._conn().execute(
            "INSERT INTO feedback (prediction_id, is_correct, true_label, created_at) VALUES (?,?,?,?)",
            (prediction_id, 1 if is_correct else 0, true_label, _now()),
        )
        self._conn().commit()
        return int(cur.lastrowid)

    def touch_user(self, device_id: str) -> str:
        conn = self._conn()
        row = conn.execute("SELECT id FROM users WHERE device_id = ?", (device_id,)).fetchone()
        if row:
            return row["id"]
        uid = uuid.uuid4().hex[:16]
        conn.execute("INSERT INTO users (id, device_id, created_at) VALUES (?,?,?)", (uid, device_id, _now()))
        conn.commit()
        return uid

    # ------------------------------------------------------------ 读
    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        n_pred = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        n_fb = conn.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()["c"]
        n_ok = conn.execute("SELECT COUNT(*) AS c FROM feedback WHERE is_correct = 1").fetchone()["c"]
        n_user = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        return {
            "predictions": n_pred,
            "feedback": n_fb,
            "feedback_correct": n_ok,
            "feedback_accuracy": round(n_ok / n_fb, 4) if n_fb else None,
            "users": n_user,
        }

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, created_at, top1_label, top1_prob, model_name, elapsed_ms"
            " FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        row = self._conn().execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["top3"] = json.loads(d.pop("top3_json"))
        return d
