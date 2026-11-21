"""E3 效率实测：参数量 / FLOPs / CPU 单张推理延迟。

为什么需要这个脚本：
    报告里要回答"手机上该选哪个模型"，而这个问题不能只靠参数量回答。
    参数量少不等于算得快 —— MobileNetV3 的深度可分离卷积在 GPU 上
    算子效率反而不如标准卷积（实测每轮比 ResNet18 慢 7%）。
    必须实测：参数量、FLOPs、CPU 延迟、内存占用。

    这个脚本不依赖 Colab、不依赖 GPU、不需要训练好的权重，
    只测架构本身的计算成本。所有模型都用随机初始化。

用法：
    python src/bench_efficiency.py
    python src/bench_efficiency.py --models resnet18,mobilenet_v3_large,swin_t
    python src/bench_efficiency.py --threads 1 4 8
    python src/bench_efficiency.py --batch-size 1 --repeat 30

输出：
    ml/artifacts/efficiency.md   人看的对比表
    ml/artifacts/efficiency.csv  机器读的原始数据

FLOPs 计算说明：
    用 torch.utils.flop_counter.FlopCounterMode（PyTorch 2.x 内置），
    比第三方 thop/fvcore 更可靠，且不需要额外安装。
    FlopCounterMode 统计的是乘加次数（MAC），1 MAC = 2 FLOPs。
    本脚本统一报 FLOPs（= 2 × MAC），并在表头注明。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from models import EXPERIMENT_MODELS, build_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def count_params(model: torch.nn.Module) -> float:
    """可训练参数量，单位百万。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_flops(model: torch.nn.Module, img_size: int = 224) -> float:
    """单张前向的 FLOPs，单位 G（= 2 × MAC）。"""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    mac = counter.get_total_flops()  # 名字叫 flops 但实际统计的是 MAC
    return mac * 2 / 1e9


def measure_latency(
    model: torch.nn.Module,
    img_size: int,
    batch_size: int,
    warmup: int,
    repeat: int,
    threads: int,
) -> dict:
    """测 CPU 推理延迟。返回单张毫秒数的中位数/均值/标准差/最小/最大。"""
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size)

    # 预热：让 PyTorch 完成算子选择、内存分配
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)

    # 换算成单张毫秒
    per_img = [t / batch_size * 1000 for t in times]
    return {
        "batch_size": batch_size,
        "threads": threads,
        "latency_ms_mean": statistics.mean(per_img),
        "latency_ms_median": statistics.median(per_img),
        "latency_ms_std": statistics.stdev(per_img) if len(per_img) > 1 else 0.0,
        "latency_ms_min": min(per_img),
        "latency_ms_max": max(per_img),
        "img_per_s": batch_size / statistics.median(times),
    }


def bench_one(
    name: str,
    img_size: int,
    batch_size: int,
    warmup: int,
    repeat: int,
    threads: int,
    num_classes: int,
) -> dict:
    print(f"\n[{name}] 构建模型…", flush=True)
    model, meta = build_model(name, num_classes, pretrained=False)
    model.eval()

    params = count_params(model)
    print(f"[{name}] 参数量 {params:.3f} M，算 FLOPs…", flush=True)
    flops = count_flops(model, img_size)
    print(f"[{name}] FLOPs {flops:.3f} G，测延迟（threads={threads}）…", flush=True)

    lat = measure_latency(model, img_size, batch_size, warmup, repeat, threads)

    row = {
        "model": name,
        "family": meta.family,
        "params_m": round(params, 3),
        "flops_g": round(flops, 3),
        "img_size": img_size,
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in lat.items()},
    }
    print(
        f"[{name}] 单张 {row['latency_ms_median']:.1f} ms "
        f"(中位数) / {row['img_per_s']:.1f} img/s",
        flush=True,
    )
    return row


def to_markdown(rows: list[dict], threads: int, batch_size: int, img_size: int) -> str:
    lines = [
        f"# 模型效率实测（E3）",
        "",
        f"- 设备：{platform.processor() or platform.machine()}",
        f"- 平台：{platform.system()} {platform.release()} / Python {platform.python_version()}",
        f"- PyTorch：{torch.__version__}",
        f"- 输入：{img_size}×{img_size}，batch={batch_size}，线程={threads}",
        f"- 延迟统计：预热后重复测量取中位数（单张毫秒）",
        f"- FLOPs 口径：1 MAC = 2 FLOPs，单张前向",
        "",
        "| 模型 | 家族 | 参数量(M) | FLOPs(G) | 单张延迟(ms) | 吞吐(img/s) | 延迟标准差(ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['family']} | {r['params_m']:.3f} | {r['flops_g']:.3f} "
            f"| {r['latency_ms_median']:.1f} | {r['img_per_s']:.1f} | {r['latency_ms_std']:.1f} |"
        )

    # 相对 ResNet18 的倍数（报告里最好用的对比方式）
    base = next((r for r in rows if r["model"] == "resnet18"), None)
    if base:
        lines += [
            "",
            "## 相对 ResNet18 的倍数",
            "",
            "| 模型 | 参数量 | FLOPs | 延迟 |",
            "|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['model']} "
                f"| {r['params_m'] / base['params_m']:.2f}× "
                f"| {r['flops_g'] / base['flops_g']:.2f}× "
                f"| {r['latency_ms_median'] / base['latency_ms_median']:.2f}× |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 效率实测")
    ap.add_argument("--models", default=",".join(EXPERIMENT_MODELS))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=1, help="模拟线上单张请求用 1")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 表示用 PyTorch 默认")
    ap.add_argument("--num-classes", type=int, default=38)
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    args = ap.parse_args()

    threads = args.threads or torch.get_num_threads()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 62)
    print(f"  E3 效率实测")
    print(f"  设备 {platform.processor() or platform.machine()}")
    print(f"  PyTorch {torch.__version__}  线程 {threads}")
    print(f"  输入 {args.img_size}×{args.img_size}  batch {args.batch_size}")
    print("=" * 62)

    rows = []
    for name in models:
        try:
            rows.append(
                bench_one(
                    name,
                    args.img_size,
                    args.batch_size,
                    args.warmup,
                    args.repeat,
                    threads,
                    args.num_classes,
                )
            )
        except Exception as e:  # 一个模型失败不影响其他
            print(f"[{name}] 失败：{type(e).__name__}: {e}", flush=True)

    if not rows:
        print("没有任何结果")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = to_markdown(rows, threads, args.batch_size, args.img_size)
    (out_dir / "efficiency.md").write_text(md, encoding="utf-8")

    keys = list(rows[0].keys())
    with (out_dir / "efficiency.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # 附一份环境信息，报告里要写测试条件
    env = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": threads,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }
    (out_dir / "efficiency_env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(md)
    print(f"[saved] {out_dir / 'efficiency.md'}")
    print(f"[saved] {out_dir / 'efficiency.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
