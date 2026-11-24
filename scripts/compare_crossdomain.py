"""汇总四个模型的同域/跨域对比，输出 Markdown 表格。

用法：
    python scripts/compare_crossdomain.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "ml" / "runs"
OUT = ROOT / "ml" / "artifacts" / "crossdomain_compare.md"


def load(run: str, fname: str) -> dict | None:
    p = RUNS / run / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    runs = sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    rows = []
    for run in runs:
        s = load(run, "summary.json") or load(run, "summary_eval.json")
        c = load(run, "crossdomain.json")
        if not s or not c:
            continue
        rows.append(
            {
                "run": run,
                "model": s.get("model", "?"),
                "params": s.get("params_m", 0),
                "test_acc": s.get("test_acc", 0),
                "macro_f1": s.get("test_macro_f1", 0),
                "cd_top1": c.get("top1_acc", 0),
                "cd_top3": c.get("top3_acc", 0),
                "cd_crop": c.get("crop_acc", 0),
                "conf": c.get("mean_confidence", 0),
                "epoch": c.get("ckpt_epoch", "-"),
            }
        )

    if not rows:
        print("没有可汇总的 run（需要同时有 summary 与 crossdomain.json）")
        return 1

    rows.sort(key=lambda r: r["cd_top1"], reverse=True)

    lines = [
        "# 同域 vs 跨域对比",
        "",
        "> 同域 = PlantVillage test（10878 张）；跨域 = PlantDoc（2920 张，27 类映射，不做任何微调）",
        "",
        "| run | 模型 | 参数量(M) | 同域 top-1 | 同域 macro-F1 | 跨域 top-1 | 跨域 top-3 | 跨域作物级 | 平均置信度 | 落差(pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        gap = (r["test_acc"] - r["cd_top1"]) * 100
        lines.append(
            f"| {r['run']} | {r['model']} | {r['params']:.3f} | {r['test_acc']:.2%} | {r['macro_f1']:.2%} | "
            f"{r['cd_top1']:.2%} | {r['cd_top3']:.2%} | {r['cd_crop']:.2%} | {r['conf']:.3f} | {gap:.1f} |"
        )

    lines += ["", "## 关键观察", ""]

    # 同域最优 vs 跨域最优
    best_dom = max(rows, key=lambda r: r["test_acc"])
    best_cd = max(rows, key=lambda r: r["cd_top1"])
    lines.append(
        f"- **同域最优**：{best_dom['model']}（{best_dom['test_acc']:.2%}）；"
        f"**跨域最优**：{best_cd['model']}（{best_cd['cd_top1']:.2%}）"
    )
    if best_dom["model"] != best_cd["model"]:
        lines.append("  → 两个排名不一致，说明同域精度不能预测跨域表现。")

    # 参数量 vs 跨域
    lines.append("")
    lines.append("| run | 参数量(M) | 跨域 top-1 | 每百万参数换来的跨域精度 |")
    lines.append("|---|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x["params"]):
        ratio = r["cd_top1"] / r["params"] * 100 if r["params"] else 0
        lines.append(f"| {r['run']} | {r['params']:.3f} | {r['cd_top1']:.2%} | {ratio:.2f} |")

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"已写出 {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
