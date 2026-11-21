"""在 Colab 会话里排队跑多个模型，一次启动、无人值守。

为什么需要：
    `colab_launch.py` 一次只起一个训练进程。要在 ResNet18 之后接着跑
    MobileNetV3 和 Swin-T，就得人盯着上一个结束再手动起下一个 ——
    中间那段空档（尤其夜里）纯属浪费 Colab 时长，而且一旦忘了就白等。

    这个脚本把整个队列交给远端的一个 bash 循环，本地断线也不影响。
    每个模型独立写日志（/content/logs/<run_name>.log），互不覆盖。

用法：
    python scripts/colab_chain.py --session train --models resnet18,mobilenet_v3_large,swin_t
    python scripts/colab_chain.py --session train --models mobilenet_v3_large,swin_t --skip-first
    python scripts/colab_chain.py --session train --status          # 看队列整体进度
    python scripts/colab_chain.py --session train --tail            # 看当前模型日志

设计要点：
    - 每个模型跑完检查 runs/<run_name>/summary.json 是否存在，据此判定成功/失败；
    - 一个模型失败不中断整个队列（继续跑下一个），最后汇总；
    - 队列结束写 /content/queue.done，内容是每个模型的状态，便于轮询。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "ml" / "src" / "train_colab.py"
REMOTE_DIR = "/content/work"
REMOTE_LOGS = "/content/logs"
REMOTE_QUEUE_DONE = "/content/queue.done"
REMOTE_QUEUE_PID = "/content/queue.pid"
REMOTE_CURRENT = "/content/queue.current"

# 每个模型的最佳 batch size：Swin-T 显存占用大，用 32 更稳
MODEL_BATCH = {
    "resnet18": 64,
    "resnet50": 48,
    "mobilenet_v3_large": 64,
    "mobilenet_v3_small": 96,
    "shufflenet_v2_x1_0": 96,
    "efficientnet_b0": 64,
    "swin_t": 32,
    "vit_b_16": 32,
}

# 每个模型的最佳学习率：Transformer 类需要更小的 lr 才稳
MODEL_LR = {
    "resnet18": "3e-4",
    "resnet50": "3e-4",
    "mobilenet_v3_large": "3e-4",
    "mobilenet_v3_small": "3e-4",
    "shufflenet_v2_x1_0": "3e-4",
    "efficientnet_b0": "3e-4",
    "swin_t": "2e-4",
    "vit_b_16": "2e-4",
}


def run_colab(args: list[str], stdin_text: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["colab"] + args,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def exec_code(code: str, session: str, timeout: int = 300) -> tuple[int, str]:
    p = run_colab(["exec", "-s", session, "--timeout", "60"], stdin_text=code, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def launch_chain(session: str, models: list[str], epochs: int, img_size: int, num_workers: int) -> int:
    """把整个队列交给远端 bash 循环。"""
    # 1. 上传训练脚本
    exec_code(f"import os; os.makedirs('{REMOTE_DIR}', exist_ok=True); os.makedirs('{REMOTE_LOGS}', exist_ok=True); print('ok')", session)
    p = run_colab(["upload", "-s", session, str(SCRIPT), f"{REMOTE_DIR}/train_colab.py"], timeout=300)
    if p.returncode != 0:
        print("上传失败：", (p.stderr or p.stdout)[-800:])
        return 1

    # 2. 生成远端 bash 脚本：逐个模型跑，失败继续
    lines = [
        "#!/bin/bash",
        f"cd {REMOTE_DIR}",
        "export PYTHONUTF8=1",
        "rm -f " + REMOTE_QUEUE_DONE,
        "RESULT=''",
    ]
    for m in models:
        run_name = f"{m}_{img_size}_{epochs}ep"
        bs = MODEL_BATCH.get(m, 64)
        lr = MODEL_LR.get(m, "3e-4")
        lines += [
            f"echo '{run_name}' > {REMOTE_CURRENT}",
            f"echo '[chain] ===== 开始 {run_name} (bs={bs} lr={lr}) ====='",
            f"COLAB_MODEL={m} COLAB_EPOCHS={epochs} COLAB_IMG_SIZE={img_size} "
            f"COLAB_BATCH_SIZE={bs} COLAB_NUM_WORKERS={num_workers} COLAB_LR={lr} "
            f"COLAB_STRONG_AUG=1 COLAB_RUN_NAME={run_name} COLAB_BENCH=0 COLAB_PREPARE_ONLY=0 "
            f"python -u train_colab.py > {REMOTE_LOGS}/{run_name}.log 2>&1",
            "RC=$?",
            f"if [ -f /content/runs/{run_name}/summary.json ]; then",
            f"  RESULT=\"$RESULT{run_name}=OK \"",
            "else",
            f"  RESULT=\"$RESULT{run_name}=FAIL(rc=$RC) \"",
            "fi",
            "echo \"[chain] $RESULT\"",
        ]
    lines += [
        f"echo \"$RESULT\" > {REMOTE_QUEUE_DONE}",
        f"rm -f {REMOTE_CURRENT}",
        "echo '[chain] 队列全部结束'",
    ]
    bash = "\n".join(lines)

    # 把 bash 脚本内容用 base64 传过去，避免引号转义问题
    import base64

    b64 = base64.b64encode(bash.encode("utf-8")).decode("ascii")
    code = f"""
import subprocess, os, base64
bash = base64.b64decode('{b64}').decode('utf-8')
with open('/content/work/chain.sh', 'w') as f:
    f.write(bash)
os.chmod('/content/work/chain.sh', 0o755)
try:
    os.remove('{REMOTE_QUEUE_DONE}')
except FileNotFoundError:
    pass
with open('/content/chain.log', 'w') as logf:
    proc = subprocess.Popen(['bash', '/content/work/chain.sh'],
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
open('{REMOTE_QUEUE_PID}', 'w').write(str(proc.pid))
print('QUEUE_LAUNCHED pid=' + str(proc.pid))
"""
    rc, out = exec_code(code, session)
    print(out.strip()[-600:])
    if "QUEUE_LAUNCHED" not in out:
        print("队列启动失败")
        return 1
    print(f"\n队列已启动：{models}")
    print(f"  队列日志 /content/chain.log，各模型日志 {REMOTE_LOGS}/<run_name>.log")
    return 0


def status(session: str) -> int:
    code = f"""
import os, json
info = {{}}
info['queue_done'] = os.path.exists('{REMOTE_QUEUE_DONE}')
if info['queue_done']:
    info['result'] = open('{REMOTE_QUEUE_DONE}').read().strip()
info['current'] = open('{REMOTE_CURRENT}').read().strip() if os.path.exists('{REMOTE_CURRENT}') else None
try:
    pid = int(open('{REMOTE_QUEUE_PID}').read().strip())
    try:
        os.kill(pid, 0); info['alive'] = True
    except OSError:
        info['alive'] = False
except Exception:
    info['alive'] = None
# 已完成哪些 run
runs = []
rd = '/content/runs'
if os.path.isdir(rd):
    for d in sorted(os.listdir(rd)):
        p = os.path.join(rd, d)
        if os.path.isdir(p):
            runs.append({{'run': d, 'summary': os.path.exists(os.path.join(p, 'summary.json'))}})
info['runs'] = runs
# 当前模型日志尾部
cur = info['current']
if cur:
    lp = f'{REMOTE_LOGS}/' + cur + '.log'
    if os.path.exists(lp):
        ls = open(lp, encoding='utf-8', errors='replace').readlines()
        info['last'] = [l.rstrip() for l in ls[-5:]]
print('QUEUE_STATUS ' + json.dumps(info, ensure_ascii=False))
"""
    rc, out = exec_code(code, session)
    print(out)
    return rc


def tail(session: str, lines: int) -> int:
    code = f"""
import os
cur = open('{REMOTE_CURRENT}').read().strip() if os.path.exists('{REMOTE_CURRENT}') else None
print('当前模型:', cur)
p = f'{REMOTE_LOGS}/' + cur + '.log' if cur else '/content/chain.log'
if os.path.exists(p):
    ls = open(p, encoding='utf-8', errors='replace').readlines()
    print(''.join(ls[-{lines}:]))
    print('--- 总行数', len(ls))
else:
    print('日志不存在:', p)
"""
    rc, out = exec_code(code, session)
    print(out)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="在 Colab 上排队跑多个模型")
    ap.add_argument("--session", default="train")
    ap.add_argument("--models", default="resnet18,mobilenet_v3_large,swin_t", help="逗号分隔")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--lines", type=int, default=40)
    args = ap.parse_args()

    if args.status:
        return status(args.session)
    if args.tail:
        return tail(args.session, args.lines)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_BATCH]
    if unknown:
        print(f"未知模型：{unknown}，可用：{sorted(MODEL_BATCH)}", file=sys.stderr)
        return 1
    return launch_chain(args.session, models, args.epochs, args.img_size, args.num_workers)


if __name__ == "__main__":
    sys.exit(main())
