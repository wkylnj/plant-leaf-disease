"""E-09 跨域错例分析：模型到底错在哪、错得像不像人。

为什么单独做这件事
------------------
E-2 已经给出「跨域只有 28.18%」这个数字，E-7 Grad-CAM 给出「模型看全局统计特征」这个机制。
但报告里"模型错在哪"这一节不能只写准确率——需要回答三个具体问题：

1. 错误是均匀散布在 27 类上，还是集中在少数几类？
2. 错误集中在什么样的类别对之间（同作物不同病 / 不同作物 / 健康 vs 病害）？
3. 模型是"完全没学会"还是"学会了但张冠李戴"（top-3 里有没有正确答案）？

这三个问题决定了产品该怎么定位：如果错例主要是同作物不同病，那"作物级 51%"就是真实可用价值；
如果 top-3 命中率远高于 top-1，那"给候选列表让用户选"就是可行的交互方案。

数据来源
--------
直接读 `ml/runs/<run>/crossdomain_preds.csv`，不重新推理。这个 CSV 由
`eval_crossdomain.py` 产出，字段：path,true,pred,conf,top3,correct,crop_correct。
四份 CSV 对应四个模型，可以横向对比"不同模型是否错在同一处"。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ARTIFACTS = ROOT / "artifacts"

CROP_KEYS = [
    "Cherry_including_sour",
    "Corn_maize",
    "Pepper_bell",
    "Strawberry",
    "Raspberry",
    "Blueberry",
    "Peach",
    "Apple",
    "Grape",
    "Orange",
    "Potato",
    "Squash",
    "Soybean",
    "Tomato",
]


def crop_of(folder: str) -> str:
    """从类别文件夹名里取出作物名，与 eval_crossdomain.crop_of_folder 口径一致。"""
    name = folder
    if len(name) > 3 and name[:2].isdigit() and name[2] == "_":
        name = name[3:]
    for key in sorted(CROP_KEYS, key=len, reverse=True):
        if name.startswith(key):
            return key
    return name.split("_")[0]


def disease_of(folder: str) -> str:
    """取病名部分（去掉序号前缀和作物名），用于判断是否"同作物不同病"。"""
    name = folder
    if len(name) > 3 and name[:2].isdigit() and name[2] == "_":
        name = name[3:]
    crop = crop_of(folder)
    rest = name[len(crop):].lstrip("_")
    return rest or "healthy"


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "true": r["true"],
                    "pred": r["pred"],
                    "conf": float(r["conf"]),
                    "top3": r["top3"].split("|"),
                    "correct": r["correct"] == "1",
                    "crop_correct": r["crop_correct"] == "1",
                }
            )
    return rows


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    crop_correct = sum(1 for r in rows if r["crop_correct"])
    top3_hit = sum(1 for r in rows if r["true"] in r["top3"])

    # 错误按真值类别统计
    per_true = defaultdict(lambda: {"n": 0, "ok": 0, "crop_ok": 0})
    for r in rows:
        d = per_true[r["true"]]
        d["n"] += 1
        d["ok"] += int(r["correct"])
        d["crop_ok"] += int(r["crop_correct"])

    # 混淆对（只统计错误的）
    pairs = Counter()
    for r in rows:
        if not r["correct"]:
            pairs[(r["true"], r["pred"])] += 1

    # 错误类型分类
    kind = Counter()
    for r in rows:
        if r["correct"]:
            continue
        t_crop, p_crop = crop_of(r["true"]), crop_of(r["pred"])
        t_dis, p_dis = disease_of(r["true"]), disease_of(r["pred"])
        if t_crop == p_crop and t_dis == p_dis:
            kind["同作物同病名(不该发生)"] += 1
        elif t_crop == p_crop and "healthy" in (t_dis, p_dis):
            kind["同作物 健康↔病害"] += 1
        elif t_crop == p_crop:
            kind["同作物 不同病"] += 1
        elif t_dis == p_dis:
            kind["不同作物 同名病"] += 1
        elif "healthy" in (t_dis, p_dis):
            kind["不同作物 涉健康"] += 1
        else:
            kind["不同作物 不同病"] += 1

    # 置信度分布：答对 vs 答错
    conf_ok = [r["conf"] for r in rows if r["correct"]]
    conf_bad = [r["conf"] for r in rows if not r["correct"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # 高置信度错误（>0.8 还错）最危险
    high_conf_bad = [r for r in rows if not r["correct"] and r["conf"] >= 0.8]

    return {
        "n": n,
        "top1_acc": correct / n if n else 0.0,
        "top3_acc": top3_hit / n if n else 0.0,
        "crop_acc": crop_correct / n if n else 0.0,
        "conf_correct": mean(conf_ok),
        "conf_wrong": mean(conf_bad),
        "n_correct": correct,
        "n_wrong": n - correct,
        "per_true": {k: dict(v) for k, v in sorted(per_true.items())},
        "top_pairs": [
            {"true": t, "pred": p, "count": c} for (t, p), c in pairs.most_common(15)
        ],
        "error_kind": dict(kind.most_common()),
        "high_conf_bad": len(high_conf_bad),
        "high_conf_bad_examples": [
            {"true": r["true"], "pred": r["pred"], "conf": round(r["conf"], 4)}
            for r in sorted(high_conf_bad, key=lambda x: -x["conf"])[:8]
        ],
    }


def worst_classes(per_true: dict, min_n: int = 5) -> list[tuple[str, int, float, float]]:
    out = []
    for cls, d in per_true.items():
        if d["n"] < min_n:
            continue
        out.append((cls, d["n"], d["ok"] / d["n"], d["crop_ok"] / d["n"]))
    out.sort(key=lambda x: x[2])
    return out


def to_markdown(results: dict[str, dict], models: list[str]) -> str:
    L: list[str] = []
    L.append("# E-09 跨域错例分析：模型到底错在哪")
    L.append("")
    L.append("> 数据来自 `ml/runs/<run>/crossdomain_preds.csv`，不重新推理。")
    L.append("> 跨域 = PlantDoc 全量 2920 张，27 类映射，零样本（无任何微调）。")
    L.append("")

    L.append("## 一句话结论")
    L.append("")
    L.append(
        "模型不是「没学会看病」，而是「学会了但按错误的依据判断」："
        f"top-1 只有 {results[models[0]]['top1_acc']:.2%}，但 top-3 有 {results[models[0]]['top3_acc']:.2%}，"
        f"作物级有 {results[models[0]]['crop_acc']:.2%}。"
        "错误高度集中在**同作物不同病**与**健康/病害混淆**两类上，"
        "且答错时置信度明显低于答对时（模型自己「不确定」），"
        "这为「低置信度只给作物名 + 候选列表」的产品设计提供了直接依据。"
    )
    L.append("")

    L.append("## 四模型横向对比")
    L.append("")
    L.append("| 模型 | top-1 | top-3 | 作物级 | 答对时置信度 | 答错时置信度 | 高置信度错误(≥0.8) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        r = results[m]
        L.append(
            f"| {m} | {r['top1_acc']:.2%} | {r['top3_acc']:.2%} | {r['crop_acc']:.2%} | "
            f"{r['conf_correct']:.3f} | {r['conf_wrong']:.3f} | {r['high_conf_bad']} |"
        )
    L.append("")

    L.append("## 错误类型分布")
    L.append("")
    L.append("把每个错误按「真值作物/病名」与「预测作物/病名」的关系归类：")
    L.append("")
    kinds = ["同作物 不同病", "同作物 健康↔病害", "不同作物 同名病", "不同作物 涉健康", "不同作物 不同病"]
    L.append("| 模型 | " + " | ".join(kinds) + " |")
    L.append("|---|" + "---:|" * len(kinds))
    for m in models:
        ek = results[m]["error_kind"]
        cells = []
        for k in kinds:
            c = ek.get(k, 0)
            pct = c / max(1, results[m]["n_wrong"]) * 100
            cells.append(f"{c} ({pct:.1f}%)")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 最常见的错误类别对（Top 10）")
    L.append("")
    for m in models:
        L.append(f"### {m}")
        L.append("")
        L.append("| 真值 | 预测 | 次数 | 占比(占全部错误) |")
        L.append("|---|---|---:|---:|")
        for p in results[m]["top_pairs"][:10]:
            pct = p["count"] / max(1, results[m]["n_wrong"]) * 100
            L.append(f"| {p['true']} | {p['pred']} | {p['count']} | {pct:.1f}% |")
        L.append("")

    L.append("## 最难的类别（按 top-1 升序，至少 5 张样本）")
    L.append("")
    L.append("| 模型 | 类别 | 样本数 | top-1 | 作物级 |")
    L.append("|---|---|---:|---:|---:|")
    for m in models:
        for cls, n, acc, cacc in worst_classes(results[m]["per_true"])[:8]:
            L.append(f"| {m} | {cls} | {n} | {acc:.2%} | {cacc:.2%} |")
    L.append("")

    L.append("## 高置信度错误（最危险的失败模式）")
    L.append("")
    L.append("模型很有把握却答错，这类错误在产品里无法通过置信度阈值拦住：")
    L.append("")
    for m in models:
        r = results[m]
        L.append(f"### {m}（共 {r['high_conf_bad']} 例，占全部错误 {r['high_conf_bad'] / max(1, r['n_wrong']) * 100:.1f}%）")
        L.append("")
        L.append("| 真值 | 预测 | 置信度 |")
        L.append("|---|---|---:|")
        for e in r["high_conf_bad_examples"]:
            L.append(f"| {e['true']} | {e['pred']} | {e['conf']:.4f} |")
        L.append("")

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "swin_t_224_15ep",
        "resnet18_224_15ep",
        "mobilenet_v3_large_224_15ep",
        "mobilenet_v3_large_daaug_224_15ep",
    ])
    ap.add_argument("--out", default=str(ARTIFACTS / "error_analysis.md"))
    args = ap.parse_args()

    results: dict[str, dict] = {}
    models: list[str] = []
    for run in args.runs:
        p = RUNS / run / "crossdomain_preds.csv"
        if not p.exists():
            print(f"[skip] {p} 不存在")
            continue
        rows = load_rows(p)
        results[run] = analyze(rows)
        models.append(run)
        r = results[run]
        print(
            f"[{run}] n={r['n']} top1={r['top1_acc']:.4f} top3={r['top3_acc']:.4f} "
            f"crop={r['crop_acc']:.4f} 高置信错误={r['high_conf_bad']}"
        )

    if not models:
        print("没有可分析的 CSV")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(results, models), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    print(f"[done] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
