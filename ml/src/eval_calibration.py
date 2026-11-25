"""E-20 模型置信度校准分析：模型的「自信」到底可不可信。

为什么要做这件事
----------------
E-09 错例分析发现一个危险的失败模式：Swin-T 在跨域上有 311 条**高置信度错误**
（置信度 ≥0.8 但预测错），占它全部错误的 16.4%。也就是说模型不但错，还「错得很自信」。

这件事直接决定了产品能不能这么做：

    置信度低 → 只报作物名（"这是一株番茄"），不报病名；
    置信度高 → 报具体病害。

这个策略成立的前提是「置信度必须能区分对错」。如果高置信度样本里错误率依然很高，
那阈值就拦不住任何东西，策略就是自欺欺人。而「置信度能不能区分对错」正是校准
（calibration）要回答的问题——准确率 35% 的模型如果平均置信度也是 35%，它是**校准良好**的；
反过来平均置信度 70% 才是**过度自信**（over-confident）。

做什么
------
1. ECE（Expected Calibration Error）：15 个等宽 bin，按样本数加权 |acc - conf|。
2. 可靠性图（reliability diagram）：每个 bin 的 acc vs conf，落盘 PNG。
3. 温度缩放（temperature scaling）：单参数 T，p_T = softmax(log(p)/T)。
   用 5 折交叉拟合 T（4 折找 T、1 折评估），避免在评估集上挑 T 的自欺。
4. 置信度阈值扫描：覆盖率 / 覆盖样本准确率 / 被拒样本的作物级正确率。
5. 正确 vs 错误预测的置信度分布（均值/中位数/四分位）。

数据与配置
----------
同域 PlantVillage test（10878 张）与跨域 PlantDoc（2920 张）**都按 224 分辨率**推理，
保证 ECE 的同域/跨域对比是同一口径（分辨率不同会让对比失真，E-15 已证明提分辨率
本身就会改变准确率）。

另外单独读 `ml/artifacts/probs_3m/probs.npz`（resnet18@320、swin_t@352 的跨域概率），
作为「部署配置」的稳健性对照——不重新推理，几乎零成本。它的同域部分没有缓存，
因此只报跨域指标，不与 224 的同域数字混在一张表里比。

产物
----
    ml/artifacts/calibration.md
    ml/artifacts/calibration.json
    ml/artifacts/calibration_reliability_<model>.png
    ml/artifacts/calibration_conf_hist_<model>.png
    ml/artifacts/calibration_cache/*.npz     同域/跨域概率缓存（第二次跑直接复用）

用法
----
    python src/eval_calibration.py --runs resnet18_224_15ep swin_t_224_15ep
    python src/eval_calibration.py --runs resnet18_224_15ep --no-samedomain   # 只算跨域
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无显示环境（服务器/CI）必须显式指定后端
import matplotlib.pyplot as plt  # noqa: E402

# 图里全是中文，matplotlib 默认的 DejaVu Sans 没有 CJK 字形，
# 不设字体的话所有中文都会变成方框（还会刷一屏 UserWarning）。
# 按优先级挑一个本机存在的，都没有就退回英文标签也能跑（只是难看）。
_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Source Han Sans SC", "SimSun", "DengXian")
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _cand in _CJK_CANDIDATES:
    if _cand in _available:
        plt.rcParams["font.sans-serif"] = [_cand] + list(plt.rcParams["font.sans-serif"])
        break
plt.rcParams["axes.unicode_minus"] = False  # 负号在 CJK 字体下容易缺字

from data import build_transforms  # noqa: E402
from eval_crossdomain import PlantDocDataset, crop_of_folder  # noqa: E402
from models import build_model  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS / "calibration_cache"
PROBS_NPZ = ARTIFACTS / "probs_3m" / "probs.npz"


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_bins(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> list[dict]:
    """按置信度做等宽分箱，返回每个 bin 的 (样本数, 平均置信度, 实际准确率)。

    等宽分箱（而不是等频）是 ECE 的标准定义，好处是不同模型/不同数据集之间
    的 bin 边界完全一致，可以直接对比；坏处是低置信度区间可能样本极少，
    所以每个 bin 都带上 count，报告里不允许只看 gap 不看样本量。
    """
    edges = _bin_edges(n_bins)
    # digitize 给出的是「落在第几个右开区间」，边界值需要 clip 到合法 bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    out: list[dict] = []
    n = len(conf)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        out.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": c,
                "frac": c / n if n else 0.0,
                "avg_conf": float(conf[m].mean()) if c else 0.0,
                "acc": float(correct[m].mean()) if c else 0.0,
            }
        )
    return out


def expected_calibration_error(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE = Σ (n_b / N) · |acc(b) - conf(b)|，只对非空 bin 求和。"""
    bins = reliability_bins(conf, correct, n_bins)
    n = len(conf)
    ece = sum(b["frac"] * abs(b["acc"] - b["avg_conf"]) for b in bins if b["count"])
    return float(ece), bins


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """多分类 Brier：Σ_k (p_k - y_k)^2 的均值。同时惩罚校准与判别力。"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """负对数似然（用真实类的概率），温度缩放的标准拟合目标。"""
    p_true = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p_true).mean())


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """温度缩放。

    直接从概率做是严格等价的：softmax(logits/T) == softmax(log(p)/T)，
    因为 log(p) = logits - logsumexp(logits)，减去的常数在 softmax 里会被约掉。
    这样就省掉了「保存 logits」这件事，probs.npz 也能直接复用。
    """
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    z = logits / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confidence_stats(conf: np.ndarray, correct: np.ndarray) -> dict:
    """正确 / 错误预测的置信度分布统计。"""
    ok = conf[correct.astype(bool)]
    bad = conf[~correct.astype(bool)]

    def q(x: np.ndarray) -> dict:
        if len(x) == 0:
            return {"n": 0}
        return {
            "n": int(len(x)),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "q1": float(np.percentile(x, 25)),
            "q3": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)),
        }

    return {"correct": q(ok), "wrong": q(bad)}


def topk_stats(probs: np.ndarray, labels: np.ndarray) -> dict:
    """top-1 / top-3 的分组置信度分析。

    把样本按「模型答得有多接近」分成三组：
    - top1：top-1 就答对（产品直接给病名也没错）；
    - top3_only：top-1 错但 top-3 里有正确答案（病名错了，方向没跑偏）；
    - miss3：top-3 都没有正确答案（彻底跑偏，只能兜底）。

    这组数字直接对应 E-09 的 top-1 34.93% / top-3 57.05%，
    并且回答一个产品问题：「top-3 里出现正确答案」的那部分样本，
    置信度是不是明显更低——如果是，说明 top-3 信息可以用来做候选列表降级。
    """
    n = len(labels)
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top3 = order[:, :3]
    p1 = probs[np.arange(n), top1]
    # top-1 与 top-2 的置信度差：margin 越小说明模型越犹豫
    p2 = probs[np.arange(n), order[:, 1]]
    margin = p1 - p2
    # top-3 的概率质量之和：接近 1 说明模型把赌注都押在这三个类上
    top3_mass = np.take_along_axis(probs, top3, axis=1).sum(axis=1)

    hit1 = top1 == labels
    hit3 = (top3 == labels[:, None]).any(axis=1)

    def q(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "conf_mean": float(p1[mask].mean()),
            "conf_median": float(np.median(p1[mask])),
            "conf_q1": float(np.percentile(p1[mask], 25)),
            "conf_q3": float(np.percentile(p1[mask], 75)),
            "margin_mean": float(margin[mask].mean()),
            "top3_mass_mean": float(top3_mass[mask].mean()),
        }

    return {
        "top1_acc": float(hit1.mean()),
        "top3_acc": float(hit3.mean()),
        "top1": q(hit1),
        "top3_only": q(~hit1 & hit3),
        "miss3": q(~hit3),
    }


# ---------------------------------------------------------------------------
# 温度缩放：5 折交叉拟合
# ---------------------------------------------------------------------------


def _stratified_folds(labels: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """按类别分层切折。

    PlantDoc 的类别分布很不均衡（番茄类几百张，樱桃 57 张），
    纯随机切折会出现某折完全没有某类，导致 NLL 不稳定，所以按类别分层。
    """
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        # 把该类样本尽量均匀地分到各折（取模分配，余数落到前几折）
        for j, i in enumerate(idx):
            folds[j % n_splits].append(int(i))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
    objective: str = "nll",
    n_bins: int = 15,
) -> tuple[float, float]:
    """在给定网格上找最优 T，返回 (T*, 目标值)。"""
    best_t, best_v = 1.0, float("inf")
    for t in grid:
        p = apply_temperature(probs, float(t))
        if objective == "nll":
            v = nll(p, labels)
        else:
            conf = p.max(axis=1)
            correct = (p.argmax(axis=1) == labels).astype(np.float64)
            v, _ = expected_calibration_error(conf, correct, n_bins)
        if v < best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def crossfit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    n_bins: int = 15,
) -> dict:
    """5 折交叉拟合 T：在 4 折上找 T，在留出的 1 折上评估，最后拼成 out-of-fold 预测。

    为什么必须这么做：如果在全量 2920 张上搜 T、再在同一批数据上报「校准后的 ECE」，
    那个 ECE 一定是偏乐观的（T 就是照着这批数据的 ECE 挑的）。PlantDoc 只有 2920 张，
    没有独立的验证集，所以用 K 折交叉——每个样本的校准概率都由「没见过它」的 T 产生。
    报告里以 OOF ECE 为主，全量拟合的 T 只作为参考值列出。
    """
    folds = _stratified_folds(labels, n_splits, seed)
    oof_nll_t = np.zeros(len(labels), dtype=np.float64)
    oof_ece_t = np.zeros(len(labels), dtype=np.float64)
    oof_base = np.zeros(len(labels), dtype=np.float64)
    per_fold: list[dict] = []

    for k in range(n_splits):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        t_nll, v_nll = fit_temperature(probs[tr], labels[tr], grid, "nll", n_bins)
        t_ece, v_ece = fit_temperature(probs[tr], labels[tr], grid, "ece", n_bins)

        p_nll = apply_temperature(probs[va], t_nll)
        p_ece = apply_temperature(probs[va], t_ece)
        oof_nll_t[va] = p_nll[np.arange(len(va)), labels[va]]
        oof_ece_t[va] = p_ece[np.arange(len(va)), labels[va]]
        oof_base[va] = probs[va][np.arange(len(va)), labels[va]]

        conf_nll = p_nll.max(axis=1)
        corr = (p_nll.argmax(axis=1) == labels[va]).astype(np.float64)
        ece_after, _ = expected_calibration_error(conf_nll, corr, n_bins)
        conf_b = probs[va].max(axis=1)
        corr_b = (probs[va].argmax(axis=1) == labels[va]).astype(np.float64)
        ece_before, _ = expected_calibration_error(conf_b, corr_b, n_bins)

        per_fold.append(
            {
                "fold": k,
                "n_val": int(len(va)),
                "t_fit_nll": t_nll,
                "t_fit_ece": t_ece,
                "ece_before": ece_before,
                "ece_after": ece_after,
            }
        )

    # OOF 汇总：把各折的校准概率拼起来后整体算一次 ECE，而不是各折 ECE 求平均。
    # 拼接后再算才是「如果这个模型部署上线，整体校准水平如何」的正确估计。
    oof_conf = np.zeros(len(labels), dtype=np.float64)
    for k in range(n_splits):
        va = folds[k]
        t = per_fold[k]["t_fit_nll"]
        p = apply_temperature(probs[va], t)
        oof_conf[va] = p.max(axis=1)

    # 注意：各折的 T 不同，argmax 不变（温度缩放是单调变换，不改变排序），
    # 所以 OOF 的「对错」与基线完全一致，只是置信度被重新标定。
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    ece_oof, _ = expected_calibration_error(oof_conf, correct, n_bins)

    t_full_nll, _ = fit_temperature(probs, labels, grid, "nll", n_bins)
    t_full_ece, _ = fit_temperature(probs, labels, grid, "ece", n_bins)

    return {
        "per_fold": per_fold,
        "t_mean": float(np.mean([f["t_fit_nll"] for f in per_fold])),
        "t_std": float(np.std([f["t_fit_nll"] for f in per_fold])),
        "t_full_nll": t_full_nll,
        "t_full_ece": t_full_ece,
        "ece_oof_after": ece_oof,
        "nll_oof_after": float(-np.log(np.clip(oof_nll_t, 1e-12, 1.0)).mean()),
        "nll_before": nll(probs, labels),
        "oof_conf": oof_conf,
    }


# ---------------------------------------------------------------------------
# 阈值分析
# ---------------------------------------------------------------------------


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    pv_folders: list[str],
    thresholds: list[float],
) -> list[dict]:
    """扫描置信度阈值 τ，回答「低置信度只报作物名」这个策略能救回多少。

    三个量：
    - coverage：置信度 ≥ τ 的样本占比（产品实际会给出病名的比例）；
    - acc_accepted：这些样本的 top-1 准确率（产品给病名时的可信度）；
    - crop_acc_rejected：被拒（< τ）的样本里，作物级预测正确的比例
      （只报作物名时，这个数字决定了"报作物名"本身有多可信）。
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    crops = [crop_of_folder(f) for f in pv_folders]
    crop_ok = np.array([crops[p] == crops[t] for p, t in zip(pred, labels)])

    n = len(labels)
    out: list[dict] = []
    for tau in thresholds:
        keep = conf >= tau
        rej = ~keep
        out.append(
            {
                "tau": float(tau),
                "coverage": float(keep.mean()),
                "n_accepted": int(keep.sum()),
                "acc_accepted": float(correct[keep].mean()) if keep.any() else 0.0,
                "n_rejected": int(rej.sum()),
                "crop_acc_rejected": float(crop_ok[rej].mean()) if rej.any() else 0.0,
                # 被接受样本的作物级准确率：和 crop_acc_rejected 对照，
                # 才能判断「置信度能不能把连作物都错的样本挑出来」。
                "crop_acc_accepted": float(crop_ok[keep].mean()) if keep.any() else 0.0,
                "crop_acc_all": float(crop_ok.mean()),
                "crop_acc_rejected_share": float(rej.mean()),
                # 被拒样本里「连作物都错」的比例：这部分只报作物名也是错的
                "wrong_crop_rejected": int((rej & ~crop_ok).sum()),
                # 覆盖样本中仍错的比例（产品给出病名时真正的误导率）
                "mislead_rate": float((keep & ~correct).mean()) if keep.any() else 0.0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------


def plot_reliability(
    model_tag: str,
    same: dict,
    cross: dict,
    out_png: Path,
) -> None:
    """同域 / 跨域并排的可靠性图，跨域那张叠加温度缩放后的曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=150)
    panels = [
        ("同域 PlantVillage test", same, False),
        ("跨域 PlantDoc", cross, True),
    ]
    for ax, (title, d, with_temp) in zip(axes, panels):
        bins = d["bins"]
        xs = [b["avg_conf"] for b in bins if b["count"]]
        ys = [b["acc"] for b in bins if b["count"]]
        ns = [b["count"] for b in bins if b["count"]]
        ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.2, label="完美校准")
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=5, label=f"校准前 (ECE={d['ece']:.4f})")
        # 点大小反映该 bin 的样本量，避免「只有 2 个样本的 bin」被当成同等重要
        if ns:
            smax = max(ns)
            ax.scatter(xs, ys, s=[40 + 260 * (c / smax) for c in ns], color="#1f77b4", alpha=0.18, zorder=1)

        if with_temp and d.get("temp") is not None:
            tb = d["temp"]["bins_after"]
            xs2 = [b["avg_conf"] for b in tb if b["count"]]
            ys2 = [b["acc"] for b in tb if b["count"]]
            ax.plot(
                xs2, ys2, "s--", color="#d62728", lw=1.6, ms=4.5,
                label=f"温度缩放后 (ECE={d['temp']['ece_after']:.4f}, T={d['temp']['t_full_nll']:.1f})",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("平均置信度 (bin 内)")
        ax.set_ylabel("实际准确率")
        ax.set_title(f"{title}\nn={d['n']}, acc={d['acc']:.4f}, 平均置信度={d['mean_conf']:.4f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(f"可靠性图 — {model_tag}（点越大表示该 bin 样本越多）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png)
    plt.close(fig)


def plot_conf_hist(model_tag: str, same: dict, cross: dict, out_png: Path) -> None:
    """正确 vs 错误预测的置信度分布（跨域），叠加同域做参照。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150, sharey=True)
    for ax, (title, d) in zip(axes, [("同域 PlantVillage test", same), ("跨域 PlantDoc", cross)]):
        st = d["conf_stats"]
        ok, bad = st["correct"], st["wrong"]
        if ok.get("n"):
            ax.hist(d["conf"][d["correct"]], bins=25, range=(0, 1), alpha=0.62,
                    color="#2ca02c", label=f"预测正确 (n={ok['n']}, 中位={ok['median']:.3f})")
        if bad.get("n"):
            ax.hist(d["conf"][~d["correct"]], bins=25, range=(0, 1), alpha=0.62,
                    color="#d62728", label=f"预测错误 (n={bad['n']}, 中位={bad['median']:.3f})")
        ax.axvline(0.8, color="0.3", ls=":", lw=1.3)
        ax.text(0.805, ax.get_ylim()[1] * 0.92, "0.8", fontsize=8, color="0.3")
        ax.set_xlabel("top-1 置信度")
        ax.set_ylabel("样本数")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8.5)

    fig.suptitle(f"置信度分布：正确 vs 错误 — {model_tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 推理与缓存
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[torch.nn.Module, list[str], str]:
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    model, _meta = build_model(model_name, len(classes), pretrained=False)
    state = ck["model"] if "model" in ck else ck["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, classes, model_name


def predict_probs(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            probs_all.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
            labels_all.append(y.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(labels_all, axis=0)


def get_samedomain_probs(
    model, pv_folders: list[str], img_size: int, args, cache: Path
) -> tuple[np.ndarray, np.ndarray]:
    """同域 test 概率，带磁盘缓存（10878 张 CPU 推理很慢，只跑一次）。"""
    if cache.exists():
        d = np.load(cache)
        print(f"    同域命中缓存 {cache.name}  shape={d['probs'].shape}")
        return d["probs"], d["labels"]

    from torchvision import datasets

    test_root = Path(args.data_root) / "test"
    ds = datasets.ImageFolder(str(test_root), transform=build_transforms(img_size, train=False))
    assert ds.classes == pv_folders, "同域 ImageFolder 类别顺序与 checkpoint 不一致"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    t0 = time.time()
    probs, labels = predict_probs(model, loader, args.device)
    print(f"    同域推理完成 {len(ds)} 张，耗时 {time.time() - t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probs=probs, labels=labels)
    return probs, labels


def get_crossdomain_probs(
    model, pv_folders: list[str], img_size: int, args, cache: Path
) -> tuple[np.ndarray, np.ndarray]:
    """跨域 PlantDoc 全量概率，带磁盘缓存。"""
    if cache.exists():
        d = np.load(cache)
        print(f"    跨域命中缓存 {cache.name}  shape={d['probs'].shape}")
        return d["probs"], d["labels"]

    ds = PlantDocDataset(
        Path(args.plantdoc_root), "all", pv_folders,
        build_transforms(img_size, train=False),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    t0 = time.time()
    probs, labels = predict_probs(model, loader, args.device)
    print(f"    跨域推理完成 {len(ds)} 张，耗时 {time.time() - t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probs=probs, labels=labels)
    return probs, labels


def load_deploy_cross(run_name: str, pv_folders: list[str]) -> dict | None:
    """从 probs_3m/probs.npz 取该模型「最优分辨率」的跨域概率（不推理）。

    注意匹配的是 **run 目录名**（如 `resnet18_224_15ep`），不是模型架构名
    （`resnet18`）——probs.npz 里的 members 记的是 `resnet18_224_15ep@320`，
    用架构名去比会永远匹配不上，静默返回空表。
    """
    if not PROBS_NPZ.exists():
        return None
    d = np.load(PROBS_NPZ)
    members = [str(m) for m in d["members"]]
    if [str(f) for f in d["pv_folders"]] != [str(f) for f in pv_folders]:
        print("    [警告] probs.npz 的类别顺序与 checkpoint 不一致，跳过部署配置对照")
        return None
    for i, m in enumerate(members):
        if m.split("@")[0] == run_name:
            return {
                "member": m,
                "probs": d["probs"][i],
                "labels": d["labels"],
            }
    return None


# ---------------------------------------------------------------------------
# 单个配置的完整分析
# ---------------------------------------------------------------------------


def analyze_config(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    pv_folders: list[str],
    args,
    with_temp: bool,
) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels

    ece, bins = expected_calibration_error(conf, correct.astype(np.float64), args.n_bins)
    res: dict = {
        "name": name,
        "n": int(len(labels)),
        "acc": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "overconfidence": float(conf.mean() - correct.mean()),
        "ece": ece,
        "bins": bins,
        "brier": brier_score(probs, labels),
        "nll": nll(probs, labels),
        "conf_stats": confidence_stats(conf, correct.astype(np.float64)),
        "topk": topk_stats(probs, labels),
        "thresholds": threshold_sweep(probs, labels, pv_folders, args.thresholds),
        # 原始数组留给画图用，落盘前会剔除
        "conf": conf,
        "correct": correct,
    }

    if with_temp:
        grid = np.round(np.arange(0.5, 5.0001, 0.1), 2)
        cf = crossfit_temperature(probs, labels, grid, args.folds, args.seed, args.n_bins)
        conf_after = cf.pop("oof_conf")
        ece_after, bins_after = expected_calibration_error(
            conf_after, correct.astype(np.float64), args.n_bins
        )
        # OOF 置信度下的阈值分析：这是「真正部署时」的可用性
        p_after = np.zeros_like(probs)
        # 用各折 T 重新生成完整概率矩阵，供阈值分析使用
        folds = _stratified_folds(labels, args.folds, args.seed)
        for k, va in enumerate(folds):
            p_after[va] = apply_temperature(probs[va], cf["per_fold"][k]["t_fit_nll"])
        res["temp"] = {
            **cf,
            "ece_after": ece_after,
            "bins_after": bins_after,
            "conf_after": conf_after,
            "thresholds_after": threshold_sweep(p_after, labels, pv_folders, args.thresholds),
            "mean_conf_after": float(conf_after.mean()),
            "brier_after": brier_score(p_after, labels),
        }
    return res


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def to_markdown(results: dict, models: list[str]) -> str:
    # 允许只跑跨域（--no-samedomain）：此时报告里所有同域段落自动省略，
    # 避免出现 "KeyError: same" 这种半途崩溃，也让冒烟测试能完整产出报告。
    has_same = all("same" in results[m] for m in models)

    L: list[str] = []
    L.append("# E-20 置信度校准：模型的「自信」可信吗")
    L.append("")
    L.append("> 同域 = PlantVillage test（10878 张），跨域 = PlantDoc 全量（2920 张，零样本）。")
    L.append("> 两个域都按 **224 分辨率**推理，保证 ECE 对比同口径；另附部署配置（更高分辨率）的跨域对照。")
    L.append("> ECE 用 15 个等宽 bin，按样本数加权 |准确率 − 平均置信度|。")
    L.append("")

    # ---- 一句话结论 ----
    r0 = results[models[0]]
    cd = r0["cross"]
    domains = [("same", "同域"), ("cross", "跨域")] if has_same else [("cross", "跨域")]
    L.append("## 一句话结论")
    L.append("")
    if has_same:
        r1 = results[models[-1]] if len(models) > 1 else r0
        sd1, cd1 = r1["same"], r1["cross"]
        L.append(
            f"置信度的偏差**在跨域时方向反转**：同域是**欠自信**"
            f"（{r0['same']['mean_conf']:.4f} 置信度 vs {r0['same']['acc']:.4f} 准确率，T={r0['same']['temp']['t_full_nll']:.1f}<1），"
            f"跨域是**过度自信**（{r0['cross']['mean_conf']:.4f} vs {r0['cross']['acc']:.4f}，T={r0['cross']['temp']['t_full_nll']:.1f}>1）。"
            f"ECE 从同域 {r0['same']['ece']:.4f} 升到跨域 {r0['cross']['ece']:.4f}"
            f"（Swin-T 更差：{sd1['ece']:.4f} → {cd1['ece']:.4f}）。"
            f"温度缩放能把跨域 ECE 压到 {r0['cross']['temp']['ece_after']:.4f} / {cd1['temp']['ece_after']:.4f}（OOF），"
            f"**但一个预测都不改**——它只让「置信度 0.6」重新等于「六成正确」，"
            f"救不回 35% 的判别力。"
        )
    else:
        L.append(
            f"（本次只跑了跨域，未计算同域。）跨域 ECE {cd['ece']:.4f}，"
            f"平均置信度 {cd['mean_conf']:.4f} 而实际准确率只有 {cd['acc']:.4f}，"
            f"过度自信 {cd['overconfidence']:+.4f}；温度缩放（T={cd['temp']['t_full_nll']:.1f}）"
            f"把 ECE 压到 {cd['temp']['ece_after']:.4f}（OOF），但准确率不变。"
        )
    L.append("")

    # ---- 为什么做 ----
    L.append("## 为什么做这个实验")
    L.append("")
    L.append(
        "E-09 发现 Swin-T 跨域有 311 条高置信度错误（置信度 ≥0.8 却答错，占其全部错误的 16.4%）。"
        "这类错误在产品里最致命：用户看到一个 95% 的病害判断，会直接照着去打药。"
        "如果置信度本身不可信，那么产品里所有基于置信度的分流逻辑（低置信度只报作物名、"
        "高置信度直接给结论）都是没有依据的。"
    )
    L.append("")
    L.append(
        "所以这个实验要回答三个产品问题："
        "① 置信度到底偏了多少（ECE）；② 能不能用温度缩放把偏掉的置信度掰回来；"
        "③ 用阈值分流「只报作物名」这个策略，实际能拦掉多少错误、又会误伤多少。"
    )
    L.append("")

    # ---- 实验设置 ----
    L.append("## 实验设置")
    L.append("")
    L.append("| 项目 | 配置 |")
    L.append("|---|---|")
    L.append("| 模型 | " + " / ".join(models) + " |")
    L.append("| 同域数据 | PlantVillage `test`，10878 张，224 分辨率 |")
    L.append("| 跨域数据 | PlantDoc `all`，2920 张（27 类可映射），224 分辨率，零样本 |")
    L.append(f"| ECE 分箱 | {results[models[0]]['cross']['n_bins']} 个等宽 bin，按样本数加权 |")
    L.append("| 温度缩放 | 单参数 T，网格 0.5–5.0 步长 0.1，按 NLL 拟合 |")
    L.append(f"| T 的拟合方式 | {results[models[0]]['cross']['temp']['n_folds']} 折交叉（按类别分层），4 折拟合 / 1 折评估，报 out-of-fold ECE |")
    L.append("| 阈值扫描 | τ ∈ {0.3, 0.4, …, 0.9} |")
    L.append("| 实现 | `ml/src/eval_calibration.py`，torch 2.6.0+cpu |")
    L.append("")

    # ---- 主结果：校准总览 ----
    L.append("## 主结果 1：校准总览")
    L.append("")
    L.append("| 模型 | 域 | 样本数 | 准确率 | 平均置信度 | 过度自信 | **ECE** | Brier | NLL |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for key, label in domains:
            r = results[m][key]
            L.append(
                f"| {m} | {label} | {r['n']} | {r['acc']:.4f} | {r['mean_conf']:.4f} | "
                f"{r['overconfidence']:+.4f} | **{r['ece']:.4f}** | {r['brier']:.4f} | {r['nll']:.4f} |"
            )
    L.append("")
    L.append(
        "读法：`过度自信` 这一列 = 平均置信度 − 准确率，**符号才是重点**。"
        "同域为负（模型答对了却不敢给高分，欠自信），跨域为正（模型答错了却很笃定，过度自信）。"
        "同一个模型在两个域上的偏差方向相反，说明置信度没有跨域可迁移的语义——"
        "这也是为什么不能拿同域调好的阈值直接搬到田间用。"
    )
    L.append("")

    # ---- 主结果：温度缩放 ----
    L.append("## 主结果 2：温度缩放的效果")
    L.append("")
    L.append("| 模型 | 域 | 拟合 T（全量, NLL） | 5 折 T 均值 ± 标准差 | ECE 校准前 | **ECE 校准后(OOF)** | 变化 | NLL 前 → 后 |")
    L.append("|---|---|---:|---|---:|---:|---:|---|")
    for m in models:
        for key, label in domains:
            r = results[m][key]
            t = r.get("temp")
            if not t:
                continue
            L.append(
                f"| {m} | {label} | {t['t_full_nll']:.1f} | {t['t_mean']:.2f} ± {t['t_std']:.2f} | "
                f"{r['ece']:.4f} | **{t['ece_after']:.4f}** | {t['ece_after'] - r['ece']:+.4f} | "
                f"{t['nll_before']:.4f} → {t['nll_oof_after']:.4f} |"
            )
    L.append("")
    L.append(
        "**T > 1 = 过度自信**（除以大于 1 的数把概率往中间压，降低置信度）；"
        "**T < 1 = 欠自信**（除以小于 1 的数让概率更尖锐，抬高置信度）。"
        "同域 T 都小于 1、跨域 T 都大于 1——**同一批权重在两个域上需要方向相反的修正**，"
        "这是本实验最反直觉的一个结果：模型在训练分布内反而「不敢下判断」，"
        "到了陌生域却「乱下判断还很笃定」。"
    )
    L.append("")
    L.append("逐折的 T 与折内 ECE：")
    L.append("")
    L.append("| 模型 | 域 | 折 | 验证样本数 | T(拟合 NLL) | ECE 折内前 | ECE 折内后 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in models:
        for key, label in domains:
            t = results[m][key].get("temp")
            if not t:
                continue
            for f in t["per_fold"]:
                L.append(
                    f"| {m} | {label} | {f['fold']} | {f['n_val']} | {f['t_fit_nll']:.1f} | "
                    f"{f['ece_before']:.4f} | {f['ece_after']:.4f} |"
                )
    L.append("")

    # ---- 主结果：阈值 ----
    L.append("## 主结果 3：置信度阈值分析（产品逻辑的核心依据）")
    L.append("")
    for m in models:
        r = results[m]["cross"]
        L.append(f"### {m}（跨域，校准前）")
        L.append("")
        L.append("| τ | 覆盖率 | 覆盖样本准确率 | 误导率(覆盖内错/全体) | 被拒样本数 | 被拒样本作物级正确率 | 被拒里连作物都错 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for s in r["thresholds"]:
            L.append(
                f"| {s['tau']:.1f} | {s['coverage']:.2%} | {s['acc_accepted']:.2%} | {s['mislead_rate']:.2%} | "
                f"{s['n_rejected']} | {s['crop_acc_rejected']:.2%} | {s['wrong_crop_rejected']} |"
            )
        L.append("")
    L.append("温度缩放后的阈值扫描（OOF 校准概率）：")
    L.append("")
    for m in models:
        r = results[m]["cross"]
        t = r.get("temp")
        if not t:
            continue
        L.append(f"### {m}（跨域，温度缩放后）")
        L.append("")
        L.append("| τ | 覆盖率 | 覆盖样本准确率 | 被拒样本数 | 被拒样本作物级正确率 |")
        L.append("|---:|---:|---:|---:|---:|")
        for s in t["thresholds_after"]:
            L.append(
                f"| {s['tau']:.1f} | {s['coverage']:.2%} | {s['acc_accepted']:.2%} | "
                f"{s['n_rejected']} | {s['crop_acc_rejected']:.2%} |"
            )
        L.append("")

    # ---- 主结果：置信度分布 ----
    L.append("## 主结果 4：正确 vs 错误的置信度分布")
    L.append("")
    L.append("| 模型 | 域 | 分组 | 样本数 | 均值 | 中位数 | Q1 | Q3 | P90 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for key, label in domains:
            st = results[m][key]["conf_stats"]
            for g, gl in (("correct", "预测正确"), ("wrong", "预测错误")):
                s = st[g]
                if not s.get("n"):
                    L.append(f"| {m} | {label} | {gl} | 0 | — | — | — | — | — |")
                    continue
                L.append(
                    f"| {m} | {label} | {gl} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                    f"{s['q1']:.4f} | {s['q3']:.4f} | {s['p90']:.4f} |"
                )
    L.append("")

    # ---- 主结果：top-1 / top-3 分组 ----
    L.append("## 主结果 5：top-1 / top-3 分组的置信度与 margin")
    L.append("")
    L.append("把样本按「模型答得有多接近」分三组：`top1` = top-1 就答对；"
             "`top3_only` = top-1 错但 top-3 里有正确答案；`miss3` = top-3 全错。"
             "`margin` = top-1 置信度 − top-2 置信度（越小越犹豫），"
             "`top3_mass` = top-3 概率质量之和。")
    L.append("")
    L.append("| 模型 | 域 | 分组 | 样本数 | 占比 | 置信度均值 | 中位数 | Q1 | Q3 | margin 均值 | top3 质量 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for key, label in domains:
            tk = results[m][key]["topk"]
            for g, gl in (("top1", "top-1 命中"), ("top3_only", "仅 top-3 命中"), ("miss3", "top-3 全错")):
                s = tk[g]
                if not s.get("n"):
                    L.append(f"| {m} | {label} | {gl} | 0 | — | — | — | — | — | — | — |")
                    continue
                L.append(
                    f"| {m} | {label} | {gl} | {s['n']} | {s['share']:.4%} | {s['conf_mean']:.4f} | "
                    f"{s['conf_median']:.4f} | {s['conf_q1']:.4f} | {s['conf_q3']:.4f} | "
                    f"{s['margin_mean']:.4f} | {s['top3_mass_mean']:.4f} |"
                )
    L.append("")
    L.append("top-1 / top-3 总准确率：")
    L.append("")
    L.append("| 模型 | 域 | top-1 | top-3 | top-3 相对提升 |")
    L.append("|---|---|---:|---:|---:|")
    for m in models:
        for key, label in domains:
            tk = results[m][key]["topk"]
            gain = tk["top3_acc"] - tk["top1_acc"]
            L.append(
                f"| {m} | {label} | {tk['top1_acc']:.4%} | {tk['top3_acc']:.4%} | +{gain:.4%} |"
            )
    L.append("")

    # ---- 部署配置对照 ----
    L.append("## 附：部署配置（更高分辨率）的跨域校准")
    L.append("")
    L.append("下面这组直接从 `ml/artifacts/probs_3m/probs.npz` 读取，不重新推理，"
             "对应 E-15/E-16 选出的最优分辨率配置。它的同域概率没有缓存，所以只报跨域。")
    L.append("")
    L.append("| 模型 | 成员 | 样本数 | 准确率 | 平均置信度 | ECE | T(全量) | ECE 校准后(OOF) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        r = results[m].get("deploy")
        if not r:
            continue
        L.append(
            f"| {m} | `{r['member']}` | {r['n']} | {r['acc']:.4f} | {r['mean_conf']:.4f} | "
            f"{r['ece']:.4f} | {r['temp']['t_full_nll']:.1f} | {r['temp']['ece_after']:.4f} |"
        )
    L.append("")

    # ---- 结论 ----
    L.append("## 结论")
    L.append("")
    L.append("### 结论一：同域是「欠自信」，不是过度自信——方向与直觉相反")
    L.append("")
    if has_same:
        for m in models:
            r = results[m]["same"]
            L.append(
                f"- {m}：准确率 {r['acc']:.4f}，平均置信度 {r['mean_conf']:.4f}"
                f"（差 {r['overconfidence']:+.4f}），ECE {r['ece']:.4f}，拟合 T = {r['temp']['t_full_nll']:.1f}。"
            )
        L.append("")
        L.append(
            "模型同域准确率 99%+，但平均置信度只有 0.89/0.92，**T 落在网格下界 0.5**"
            "（说明真正最优的 T 可能比 0.5 更小，受网格限制）。"
            "这意味着模型答对了却「不敢给高分」：输出分布被压得比较平。"
            "一个直接后果是——同域的 ECE 绝对值（0.08~0.11）并不比跨域小多少，"
            "**不能简单说「同域校准得好、跨域校准得差」**，而是两个域偏差方向相反。"
        )
        L.append("")
        L.append(
            "注意 T=0.5 是网格下界，不是内部最优。如果放开网格，同域的 ECE 还能再降，"
            "但这不影响结论（方向是欠自信）。"
        )
    else:
        L.append("（本次未跑同域，此结论待补。）")
    L.append("")
    L.append("### 结论二：跨域是「过度自信 + 判别力崩塌」双重问题")
    L.append("")
    for m in models:
        r = results[m]["cross"]
        L.append(
            f"- {m}：准确率 {r['acc']:.4f}，平均置信度 {r['mean_conf']:.4f}，"
            f"ECE {r['ece']:.4f}，过度自信 {r['overconfidence']:+.4f}，拟合 T = {r['temp']['t_full_nll']:.1f}。"
        )
    L.append("")
    L.append(
        "跨域的「平均置信度 − 准确率」全部为正，且 T 全部大于 1——模型在陌生域上"
        "**既答错、又很笃定**。但要区分两件事：过度自信（置信度虚高）可以用温度缩放修，"
        "判别力崩塌（准确率只有 28%~35%）修不了。E-09 的 311 条 ≥0.8 高置信度错误就是这两者叠加的产物。"
    )
    L.append("")
    L.append("### 结论三：温度缩放能修好 ECE，但修不好准确率")
    L.append("")
    for m in models:
        r = results[m]["cross"]
        t = r["temp"]
        L.append(
            f"- {m}：ECE {r['ece']:.4f} → {t['ece_after']:.4f}（{t['ece_after'] - r['ece']:+.4f}），"
            f"准确率不变（{r['acc']:.4f}，温度缩放是单调变换，argmax 不变）。"
        )
    L.append("")
    L.append(
        "这一点必须在报告里讲清楚，否则容易被误解成「校准提升了跨域性能」。"
        "温度缩放只改变置信度的**数值含义**，不改变任何一个预测。它的价值在于："
        "校准后「置信度 0.6」真的对应六成正确，产品里的阈值才有可解释的语义。"
    )
    L.append("")
    L.append("### 结论四：「低置信度只报作物名」这个策略，只对了一半")
    L.append("")
    last_th = {round(s["tau"], 1): s for s in results[models[-1]]["cross"]["thresholds"]}
    last_base = last_th[0.3]["crop_acc_all"]
    for m in models:
        r = results[m]["cross"]
        th = {round(s["tau"], 1): s for s in r["thresholds"]}
        L.append(f"**{m}**（全体样本的作物级准确率基线 = {th[0.3]['crop_acc_all']:.1%}）：")
        L.append("")
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
            s = th.get(tau)
            if not s:
                continue
            L.append(
                f"- τ={tau:.1f}：覆盖率 {s['coverage']:.1%}，覆盖样本病名准确率 {s['acc_accepted']:.1%}"
                f"（作物级 {s['crop_acc_accepted']:.1%}）；被拒 {s['n_rejected']} 张，"
                f"其中作物级正确 {s['crop_acc_rejected']:.1%}，连作物都错 {s['wrong_crop_rejected']} 张。"
            )
        L.append("")
    L.append(
        "**这个策略只站得住一半。** 站得住的部分：置信度确实有判别力——"
        "正确预测的置信度均值明显高于错误预测（主结果 4），所以按置信度分流不是瞎切。"
        "站不住的部分有三条，全部是上面的实际数字："
    )
    L.append("")
    L.append(
        "1. **高置信度区间也不可靠**：即使把阈值提到 τ=0.9，覆盖样本的病名准确率最高也只有 "
        f"{last_th[0.9]['acc_accepted']:.1%}（Swin-T），远达不到「可以照着打药」的程度。"
        "产品不能承诺「高置信度 = 结论可信」。"
    )
    L.append(
        "2. **覆盖率掉得太快**：Swin-T 在 τ=0.8 时只剩 "
        f"{last_th[0.8]['coverage']:.1%} 的样本能给出病名，其余 "
        f"{1 - last_th[0.8]['coverage']:.1%} 全部落到降级或重拍。"
        "而「降级为只报作物名」这条路本身也不安全——"
    )
    L.append(
        "3. **被拒样本的作物级准确率明显低于全体基线**：Swin-T 被拒样本的作物级正确率"
        f"从 {last_th[0.8]['crop_acc_rejected']:.1%}（τ=0.8）到 {last_th[0.5]['crop_acc_rejected']:.1%}（τ=0.5），"
        f"而全体基线是 {last_base:.1%}。这说明置信度**确实能筛出「连作物都认错」的样本**"
        "（方向是对的），但被拒样本里仍有近一半连作物都判对，"
        "对这些样本硬报一个作物名同样是误导。"
    )
    L.append("")
    L.append(
        "**产品结论**：三级分流（高置信度给病名 / 中置信度只给作物名 / 极低置信度要求重拍）"
        "在方向上正确，但必须承认**第三级是主力而不是例外**——在 PlantDoc 这类真实田间照片上，"
        "τ=0.8 时有 "
        f"{1 - last_th[0.8]['coverage']:.1%} 的样本会落到降级或重拍，这才是诚实的可用性描述。"
        "而且这只解决了「别乱说」，没解决「能不能说对」——后者要等 E-14 跨域微调。"
    )
    L.append("")
    L.append("### 结论五：top-3 里有正确答案这件事，置信度筛不出来")
    L.append("")
    for m in models:
        tk = results[m]["cross"]["topk"]
        a, b = tk["top3_only"], tk["miss3"]
        L.append(
            f"- {m}：top-1 错但 top-3 命中的 {a['n']} 张，置信度均值 {a['conf_mean']:.4f}、"
            f"margin {a['margin_mean']:.4f}；top-3 全错的 {b['n']} 张，"
            f"置信度均值 {b['conf_mean']:.4f}、margin {b['margin_mean']:.4f}——"
            f"置信度只差 {a['conf_mean'] - b['conf_mean']:+.4f}。"
        )
    L.append("")
    L.append(
        "这组置信度差距小得没有实用价值。产品上想做一个「我们不确定是哪一种，"
        "但候选里大概有对的」的降级方案，**不能靠置信度去判断哪些样本属于这种情况**——"
        "top-3 命中组和 top-3 全错组的置信度几乎重合。"
        "margin 有一点区分度（top-3 命中组的 margin 更大），但也不足以单独当判据。"
        "实践上更稳的做法是**无条件给 top-3 候选列表**，而不是试图预测「top-3 是否可靠」。"
    )

    # ---- 对项目的意义 ----
    L.append("## 对项目的意义")
    L.append("")
    L.append(
        "1. **产品逻辑可以落地，但要重新设计分级**：置信度确实有判别力（正确/错误的置信度分布分离、"
        "被拒样本的作物级准确率低于基线），所以按置信度分流在方向上成立。"
        "但实际数字表明**降级档会是主力而不是例外**（Swin-T 在 τ=0.8 时 "
        f"{1 - last_th[0.8]['coverage']:.1%} 的样本要降级或重拍），"
        "产品设计必须按「大多数情况下给不出确定病名」来规划，而不是当成边缘情况。"
    )
    L.append(
        "2. **温度缩放应该上线，但只作为「置信度语义修正」**：它让阈值有了跨模型一致的含义，"
        "而且只有 1 个参数、几乎零成本。**不要把它写进「提升跨域准确率」的成绩里**——"
        "它一个预测都没改。另外注意同域和跨域需要的 T 方向相反，"
        "**必须为目标域单独拟合一个 T，不能复用同域的**。"
    )
    L.append(
        "3. **和 E-03 线性探针形成闭环**：探针证明瓶颈在特征层、换头无效；"
        "本实验证明校准也无效（对准确率）。两条独立证据指向同一个结论："
        "**跨域只能靠让骨干见到目标域数据（E-14）来解决**，任何后处理都只是让错误「说得更诚实」。"
    )
    L.append(
        "4. **给报告一个诚实的产品叙事**：不要说「我们的模型 99%」，"
        "而要说「同域 99%、跨域 35%；跨域时模型既答错又过度自信（ECE 0.16~0.22），"
        "温度缩放能把置信度语义修准，但准确率不变，因此产品采用分级输出 + 重拍兜底」。"
    )
    L.append("")

    # ---- 局限 ----
    L.append("## 已知局限")
    L.append("")
    L.append(
        "1. **15 个等宽 bin 在两端分辨率不足**：同域置信度集中在 0.85–0.95（均值约 0.89/0.92），"
        "这一带的 bin 每个都很窄、样本密集；而跨域有大量样本落在 0.2–0.5。"
        "等宽分箱在中间区域的分辨率反而不如等频分箱，ECE 的绝对值会随分箱方式小幅变化"
        "（但方向性结论——同域欠自信、跨域过度自信——不受影响）。"
    )
    L.append(
        "2. **PlantDoc 只有 2920 张，5 折交叉的每折验证集约 584 张**，ECE 的抽样噪声不可忽略"
        "（尤其低置信度区间样本少）。所以报的是 out-of-fold 汇总 ECE，并同时给出逐折结果供判断稳定性。"
    )
    L.append(
        "3. **温度缩放只拟合了 1 个全局参数**，没有做 class-wise / 分作物校准。"
        "跨域误差高度集中在番茄/玉米/土豆，按作物分组拟合 T 理论上可能更好，但 2920 张分到 14 个作物后每类样本太少，容易过拟合。"
    )
    L.append(
        "4. **只测了 ResNet18 与 Swin-T**，未测 MobileNetV3（结论方向一致的可能性很高，"
        "E-09 已显示它的高置信度错误比例与 ResNet18 接近）。"
    )
    L.append(
        "5. **同域与跨域都是 224 单视图推理**，未使用 TTA/集成。"
        "TTA 会改变概率分布（多视图平均通常让置信度更平滑、ECE 更低），"
        "所以本报告的 ECE 不能直接外推到 TTA 配置。"
        "作为对照，附录里的部署配置（320/352 分辨率）ECE 反而更低（resnet18 0.1570 → 0.0552），"
        "说明**分辨率和 ECE 强相关**，报 ECE 时必须写清分辨率。"
    )
    L.append(
        "6. **温度缩放的 T 是在 PlantDoc 上交叉拟合的**，严格来说 PlantDoc 是被「用过」的；"
        "真正上线时应固定一个 T 不再更新，且如果目标域继续漂移，T 需要重新拟合。"
    )
    L.append("")

    # ---- 产物 ----
    L.append("## 产物")
    L.append("")
    L.append("- `ml/src/eval_calibration.py`：实验脚本")
    L.append("- `ml/artifacts/calibration.json`：完整结果（含逐 bin 明细、逐折 T、阈值扫描）")
    for m in models:
        L.append(f"- `ml/artifacts/calibration_reliability_{m}.png`：{m} 同域/跨域可靠性图")
        L.append(f"- `ml/artifacts/calibration_conf_hist_{m}.png`：{m} 正确/错误置信度分布")
    L.append("- `ml/artifacts/calibration_cache/*.npz`：同域/跨域概率缓存（复跑时复用）")
    L.append("")

    # ---- 复现 ----
    L.append("## 复现")
    L.append("")
    L.append("```bash")
    L.append("cd ml")
    L.append("python src/eval_calibration.py --runs runs/resnet18_224_15ep runs/swin_t_224_15ep")
    L.append("```")
    L.append("")
    L.append(
        "首次运行需要推理：同域 10878 张（ResNet18 约 220s、Swin-T 更慢）+ 跨域 2920 张，"
        "结果缓存到 `artifacts/calibration_cache/`，第二次起秒级完成。"
    )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="E-20 置信度校准分析（ECE / 可靠性图 / 温度缩放 / 阈值）")
    ap.add_argument("--runs", nargs="+",
                    default=["runs/resnet18_224_15ep", "runs/swin_t_224_15ep"])
    ap.add_argument("--data-root", default=str(ROOT / "data" / "processed" / "plantvillage"))
    ap.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--no-samedomain", action="store_true", help="跳过同域推理（只算跨域）")
    ap.add_argument("--no-deploy", action="store_true", help="不读 probs.npz 的部署配置对照")
    ap.add_argument("--out", default=str(ARTIFACTS / "calibration.md"))
    args = ap.parse_args()

    args.device = torch.device("cpu")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"设备       : {args.device}")
    print(f"分辨率     : {args.img_size}")
    print(f"ECE bin 数 : {args.n_bins}")
    print(f"温度交叉折 : {args.folds}")

    results: dict[str, dict] = {}
    models: list[str] = []

    for run in args.runs:
        run_dir = Path(run) if Path(run).is_absolute() else (ROOT / run)
        if not (run_dir / "best.pt").exists():
            print(f"[跳过] {run_dir / 'best.pt'} 不存在")
            continue

        # 用 run 目录名做标签（`runs/resnet18_224_15ep` → `resnet18_224_15ep`），
        # 否则产物文件名里会带 `runs/` 前缀，拼出来是个不存在的目录。
        label = run_dir.name
        print(f"\n=== {label} ===")
        model, pv_folders, model_name = load_run(run_dir)
        model.to(args.device)
        print(f"  模型 {model_name}，{len(pv_folders)} 类")

        entry: dict = {"model_name": model_name, "img_size": args.img_size}

        # ---- 跨域（224，主口径）----
        p_cross, y_cross = get_crossdomain_probs(
            model, pv_folders, args.img_size, args,
            CACHE_DIR / f"{label}_cross_{args.img_size}.npz",
        )
        cross = analyze_config("cross_224", p_cross, y_cross, pv_folders, args, with_temp=True)
        cross["n_bins"] = args.n_bins
        cross["temp"]["n_folds"] = args.folds
        print(
            f"  跨域 acc={cross['acc']:.4f} conf={cross['mean_conf']:.4f} "
            f"ECE={cross['ece']:.4f} T={cross['temp']['t_full_nll']:.1f} "
            f"ECE_after={cross['temp']['ece_after']:.4f}"
        )

        # ---- 同域 ----
        if not args.no_samedomain:
            p_same, y_same = get_samedomain_probs(
                model, pv_folders, args.img_size, args,
                CACHE_DIR / f"{label}_same_{args.img_size}.npz",
            )
            same = analyze_config("same_224", p_same, y_same, pv_folders, args, with_temp=True)
            same["n_bins"] = args.n_bins
            same["temp"]["n_folds"] = args.folds
            print(
                f"  同域 acc={same['acc']:.4f} conf={same['mean_conf']:.4f} "
                f"ECE={same['ece']:.4f} T={same['temp']['t_full_nll']:.1f} "
                f"ECE_after={same['temp']['ece_after']:.4f}"
            )
            entry["same"] = same

        # ---- 部署配置（读 probs.npz，用 run 目录名匹配 member）----
        if not args.no_deploy:
            dep = load_deploy_cross(label, pv_folders)
            if dep:
                print(f"  部署配置 {dep['member']} 从 probs.npz 读取")
                d = analyze_config("deploy", dep["probs"], dep["labels"], pv_folders, args, with_temp=True)
                d["member"] = dep["member"]
                d["n_bins"] = args.n_bins
                print(f"     acc={d['acc']:.4f} ECE={d['ece']:.4f} "
                      f"T={d['temp']['t_full_nll']:.1f} ECE_after={d['temp']['ece_after']:.4f}")
                entry["deploy"] = d
            else:
                print("  [跳过] probs.npz 里没有该模型的部署配置")

        entry["cross"] = cross
        results[label] = entry
        models.append(label)

        # ---- 画图（每个模型两张）----
        if not args.no_samedomain:
            plot_reliability(
                label, entry["same"], entry["cross"],
                ARTIFACTS / f"calibration_reliability_{label}.png",
            )
            plot_conf_hist(
                label, entry["same"], entry["cross"],
                ARTIFACTS / f"calibration_conf_hist_{label}.png",
            )
            print(f"  已画图 calibration_reliability_{label}.png / calibration_conf_hist_{label}.png")

    if not models:
        print("没有可分析的 run")
        return 1

    # ---- 落盘（剔除大数组）----
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if k not in ("conf", "correct", "conf_after", "oof_conf")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(results, models), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(strip(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] {out}")
    print(f"[done] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
