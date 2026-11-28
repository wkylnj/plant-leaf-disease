"""从 Kaggle kernel 日志（JSON-lines）里提取 ALL_RESULTS_JSON 段落并存为 json。

用法：
    python scripts/extract_kernel_json.py _kaggle_out/swinlog/swinft.log -o ml/runs/swin_t_224_15ep/plantdoc_ft_cv.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_stdout(log_path: Path):
    """逐行解析 JSON-lines，产出 stdout 文本。非 JSON 行按纯文本处理。"""
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().lstrip(",").strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield raw
                continue
            data = obj.get("data")
            if isinstance(data, str):
                yield data
        else:
            yield raw


def extract(log_path: Path, begin: str, end: str) -> str | None:
    buf: list[str] = []
    inside = False
    for text in iter_stdout(log_path):
        if begin in text:
            inside = True
            continue
        if end in text:
            break
        if inside:
            buf.append(text)
    return "".join(buf) if buf else None


def main() -> int:
    ap = argparse.ArgumentParser(description="提取 kernel 日志里的 JSON 段")
    ap.add_argument("log")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--begin", default="ALL_RESULTS_JSON_BEGIN")
    ap.add_argument("--end", default="ALL_RESULTS_JSON_END")
    args = ap.parse_args()

    text = extract(Path(args.log), args.begin, args.end)
    if not text:
        print(f"[错误] 日志里没找到 {args.begin}")
        return 1
    data = json.loads(text)
    # 有些 kernel 把结果包成 {"<model>": {"raw": {...}}} 或 {"<model>": {...}}，剥一层
    if isinstance(data, dict) and len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict):
            data = only.get("raw") if isinstance(only.get("raw"), dict) else only
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {out}")

    zs = data.get("zeroshot", {})
    ft = data.get("finetuned", {})
    pd = data.get("paired_delta", {})
    print(f"模型 {data.get('model')}  ckpt_epoch {data.get('ckpt_epoch')}")
    print(f"零样本 top-1 {zs.get('top1_mean'):.4f} ± {zs.get('top1_std'):.4f}  作物级 {zs.get('crop_mean'):.4f}")
    print(f"微调   top-1 {ft.get('top1_mean'):.4f} ± {ft.get('top1_std'):.4f}  作物级 {ft.get('crop_mean'):.4f} ± {ft.get('crop_std'):.4f}")
    print(f"配对差 top-1 {pd.get('top1_mean'):+.4f} ± {pd.get('top1_std'):.4f}  作物级 {pd.get('crop_mean'):+.4f}")
    print(f"更好折数 {pd.get('folds_better')}/{pd.get('n_folds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
