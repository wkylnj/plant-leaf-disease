"""分析 Grad-CAM summary.json，输出报告需要的统计结论。"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ml" / "runs" / "resnet18_224_15ep" / "gradcam" / "summary.json"
d = json.loads(p.read_text(encoding="utf-8"))
recs = d["records"]

print(f"模型 {d['model']}  样本 {len(recs)}  抽样准确率 {d['sample_acc'] * 100:.2f}%")
print(f"中心注意力 {d['center_ratio_mean'] * 100:.1f}%   四角注意力 {d['corners_ratio_mean'] * 100:.1f}%")
print()

# 健康类 vs 病害类
def is_healthy(name: str) -> bool:
    return "healthy" in name.lower()

groups = [("健康类", [r for r in recs if is_healthy(r["class_dir"])]),
          ("病害类", [r for r in recs if not is_healthy(r["class_dir"])])]
for tag, g in groups:
    if not g:
        continue
    print(f"{tag} n={len(g):3d}  中心 {st.mean(r['center_ratio'] for r in g) * 100:5.1f}%  "
          f"四角 {st.mean(r['corners_ratio'] for r in g) * 100:4.1f}%  "
          f"置信度均值 {st.mean(r['prob'] for r in g) * 100:5.1f}%")
print()

# 九宫格的行列边际
def col_sum(r, c):
    return sum(r["grid_mass"][c + 3 * i] for i in range(3))

def row_sum(r, ri):
    return sum(r["grid_mass"][ri * 3:(ri + 1) * 3])

print("九宫格边际分布：")
for c, name in enumerate(["左列", "中列", "右列"]):
    print(f"  {name} {st.mean(col_sum(r, c) for r in recs) * 100:5.1f}%")
for ri, name in enumerate(["上排", "中排", "下排"]):
    print(f"  {name} {st.mean(row_sum(r, ri) for r in recs) * 100:5.1f}%")
print()

# 置信度与注意力集中度的关系
concentrated = [r for r in recs if max(r["grid_mass"]) > 0.2]
diffuse = [r for r in recs if max(r["grid_mass"]) <= 0.2]
print(f"注意力集中（单格>20%）: {len(concentrated)} 张，置信度均值 "
      f"{st.mean(r['prob'] for r in concentrated) * 100:.1f}%" if concentrated else "无")
print(f"注意力分散（单格<=20%）: {len(diffuse)} 张，置信度均值 "
      f"{st.mean(r['prob'] for r in diffuse) * 100:.1f}%" if diffuse else "无")
print()

# 错例
wrong = [r for r in recs if not r["correct"]]
print(f"错例 {len(wrong)} 张：")
for r in wrong:
    print(f"  真 {r['class_dir'][:32]:<32s} 预 {r['pred_class'][:28]:<28s} "
          f"p={r['prob'] * 100:5.1f}%  中心 {r['center_ratio'] * 100:4.1f}%")
print()

# 每类中心注意力排序，看哪些类最"看中心"
by_class = {}
for r in recs:
    by_class.setdefault(r["class_dir"], []).append(r)
ranked = sorted(by_class.items(), key=lambda kv: st.mean(x["center_ratio"] for x in kv[1]))
print("中心注意力最低的 6 类（注意力最偏离中心）：")
for name, g in ranked[:6]:
    print(f"  {st.mean(r['center_ratio'] for r in g) * 100:5.1f}%  {name}")
print("中心注意力最高的 6 类（注意力最集中在中心）：")
for name, g in ranked[-6:]:
    print(f"  {st.mean(r['center_ratio'] for r in g) * 100:5.1f}%  {name}")
