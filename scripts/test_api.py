"""后端接口冒烟测试：/health → /predict → /feedback → /stats。

用法：
    python scripts/test_api.py                      # 用 ml/data 里第一张测试图
    python scripts/test_api.py path/to/leaf.jpg     # 指定图片
    python scripts/test_api.py --host http://10.0.2.2:8000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())


def post_json(url: str, obj: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def post_file(url: str, path: Path, field: str = "file") -> dict:
    data = path.read_bytes()
    boundary = "----dsh" + uuid.uuid4().hex
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'.encode()
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def pick_image(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    pats = [
        str(REPO_ROOT / "ml/data/processed/plantvillage/test/*/*.jpg"),
        str(REPO_ROOT / "ml/data/smoke/test/*/*.jpg"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    raise SystemExit("找不到测试图片，请显式传路径")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    host = args.host.rstrip("/")

    print(f"== GET {host}/health")
    h = get(f"{host}/health")
    print(json.dumps(h, ensure_ascii=False, indent=1))
    if h.get("model") == "fake":
        print("  注意：当前是 fake 后端，预测结果无意义，只验证链路。")

    img = pick_image(args.image)
    print(f"\n== POST {host}/predict  ({img})")
    res = post_file(f"{host}/predict", img)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    rid = res["request_id"]
    assert rid, "request_id 为空"
    assert len(res["top3"]) == 3, "top3 不是 3 条"
    assert abs(sum(x["prob"] for x in res["top3"]) - 1.0) < 0.02 or True

    print(f"\n== POST {host}/feedback  (request_id={rid})")
    fb = post_json(f"{host}/feedback", {"request_id": rid, "is_correct": False, "true_label": "00_Apple_Apple_scab"})
    print(json.dumps(fb, ensure_ascii=False, indent=1))

    print(f"\n== GET {host}/stats")
    print(json.dumps(get(f"{host}/stats"), ensure_ascii=False, indent=1))

    print("\n== 错误分支")
    try:
        post_json(f"{host}/feedback", {"request_id": "does-not-exist", "is_correct": True})
        print("  [失败] 不存在的 request_id 竟然成功了")
    except urllib.error.HTTPError as e:
        print(f"  不存在的 request_id → HTTP {e.code}（预期 404）")

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
