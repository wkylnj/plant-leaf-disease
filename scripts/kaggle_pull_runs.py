"""取回 Kaggle kernel 输出中 runs/ 下的产物。

坑：kernel 输出目录里混着整个 PlantVillage 数据集（5.4 万张图），从头翻页要 272 页。
解法：分页 token 是 base64({"GcsPageToken": "<jobid>/output/<相对路径>"})，
      先拿 jobid，再构造指向 "<jobid>/output/runs/" 的 token 直接跳过去，
      然后用 kernels_output 带 file_pattern 只下需要的文件。

用法：
    python _pull_runs3.py <owner/slug> <本地目录> [--with-ckpt]
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT_WANTED = (
    r"(summary\.json|history\.csv|per_class_f1\.csv|config\.json|classes\.json"
    r"|curves\.png|confusion_matrix\.png|summary_eval\.json|tta_eval\.json)$"
)
CKPT_PATTERN = r"(summary\.json|history\.csv|per_class_f1\.csv|config\.json|classes\.json|curves\.png|confusion_matrix\.png|best\.pt)$"


def b64(payload: dict) -> str:
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()


def dec(tok: str) -> dict:
    try:
        return json.loads(base64.b64decode(tok).decode())
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("dest")
    ap.add_argument("--with-ckpt", action="store_true")
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    r = api.kernels_list_files(args.slug, page_size=1)
    gcs = dec(getattr(r, "next_page_token", "") or "").get("GcsPageToken", "")
    if not gcs:
        print("[失败] 拿不到 GcsPageToken")
        return 1
    jobid = gcs.split("/output/")[0]
    print(f"[job] {jobid}")

    token = b64({"GcsPageToken": f"{jobid}/output/runs/"})
    pattern = CKPT_PATTERN if args.with_ckpt else DEFAULT_WANTED

    for attempt in range(1, 5):
        try:
            files, status = api.kernels_output(
                args.slug,
                str(dest),
                file_pattern=pattern,
                force=True,
                quiet=False,
                page_token=token,
                page_size=200,
            )
            print(f"[ok] status={status} files={len(files)}")
            for f in files:
                p = Path(f)
                sz = p.stat().st_size if p.exists() else 0
                print(f"  {p.name}  {sz / 1024 / 1024:.2f} MB")
            total = sum(
                p.stat().st_size for p in dest.rglob("*") if p.is_file()
            )
            print(f"[done] {total / 1024 / 1024:.1f} MB -> {dest}")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}] {type(e).__name__}: {e}")
            time.sleep(5 * attempt)

    print("[失败] 下载失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
