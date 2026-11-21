#!/usr/bin/env python
"""把 ml/runs/ 下所有训练结果汇总成一张对比表（Markdown + CSV）。

用法：
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --runs-dir ml/runs --out ml/artifacts
    python scripts/summarize_runs.py --sort params_m

为什么需要：
    报告里的模型对比表要随实验增加反复重排。手抄 summary.json 容易抄错，
    而且一旦重跑某个模型，旧数字会留在报告里。这个脚本保证表里的每个
    数字都来自磁盘上的 summary.json，可追溯。

输出：
    ml/artifacts/runs_summary.md   人看的对比表
    ml/artifacts/runs_summary.csv  机器读的原始数据
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# summary.json 里可能缺失的字段，给个占位
FIELDS = [
    ("model", "模型", "{}"),
    ("params_m", "参数量(M)", "{:.3f}"),
    ("img_size", "分辨率", "{}"),
    ("epochs", "轮数", "{}"),
    ("pretrained", "预训练", "{}"),
    ("strong_aug", "强增强", "{}"),
    ("best_val_acc", "最佳验证精度", "{:.4f}"),
    ("test_acc", "测试精度", "{:.4f}"),
    ("test_macro_f1", "测试 macro-F1", "{:.4f}"),
    ("test_loss", "测试损失", "{:.4f}"),
    ("onnx_max_diff", "ONNX 误差", "{:.2e}"),
]


def load_runs(runs_dir: Path) -> list[dict]:
    """读每个 run 目录下的 summary.json。"""
    rows = []
    if not runs_dir.exists():
        return rows
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"跳过 {d.name}：summary.json 读不了（{e}）")
            continue
        data["run_name"] = d.name
        # 用 history.csv 的最后一轮秒数算每轮平均耗时
        hist = d / "history.csv"
        if hist.exists():
            try:
                lines = hist.read_text(encoding="utf-8").strip().splitlines()
                secs = [float(x.split(",")[7]) for x in lines[1:] if len(x.split(",")) > 7]
                if secs:
                    data["secs_per_epoch"] = round(sum(secs) / len(secs), 1)
            except (ValueError, IndexError):
                pass
        rows.append(data)
    return rows


def fmt(val, spec: str) -> str:
    """按格式串渲染，缺失值显示为 —。"""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "是" if val else "否"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def to_markdown(rows: list[dict]) -> str:
    """生成 Markdown 表格。"""
    cols = [("run_name", "运行名", "{}")] + FIELDS + [("secs_per_epoch", "每轮秒数", "{}")]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(fmt(r.get(c[0]), c[2]) for c in cols) + " |")
    return "\n".join([head, sep] + body)


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总训练结果")
    ap.add_argument("--runs-dir", default=str(ROOT / "ml" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "artifacts"))
    ap.add_argument(
        "--sort",
        default="params_m",
        choices=["params_m", "test_acc", "test_macro_f1", "run_name", "none"],
        help="排序键，none 表示按目录名",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(runs_dir)
    if not rows:
        print(f"{runs_dir} 下没有可用的 summary.json，先跑一次训练或把 Colab 产物拉回来")
        return 1

    if args.sort != "none":
        rows.sort(key=lambda r: r.get(args.sort) if r.get(args.sort) is not None else -1)

    md = to_markdown(rows)
    (out_dir / "runs_summary.md").write_text(md + "\n", encoding="utf-8")

    keys = ["run_name"] + [f[0] for f in FIELDS] + ["secs_per_epoch"]
    with open(out_dir / "runs_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    print(f"共 {len(rows)} 个 run，已写：")
    print(f"  {out_dir / 'runs_summary.md'}")
    print(f"  {out_dir / 'runs_summary.csv'}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
