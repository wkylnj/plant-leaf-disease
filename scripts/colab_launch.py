"""在 Colab 会话里以「后台进程 + 日志文件」的方式启动训练。

为什么不用 `colab exec -f train_colab.py` 直接跑：
`colab exec` 靠一条长连接把远端输出流回来，连接一断（笔记本休眠、网络抖动、
本地进程被误杀）输出就没了，虽然远端 kernel 还在算，但我们看不到结果、
也拿不到最终产物。改成 nohup 分离进程后：
  - 本地断线不影响远端训练；
  - 随时用 `colab exec` 读日志尾部看进度；
  - 训练结束写一个 .done 标记，轮询脚本据此判断。

用法：
    python scripts/colab_launch.py                     # 用默认参数启动
    python scripts/colab_launch.py --epochs 15 --model resnet18
    python scripts/colab_launch.py --tail              # 只看当前日志尾部
    python scripts/colab_launch.py --status            # 看是否还在跑
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
REMOTE_LOG = "/content/train.log"
REMOTE_DONE = "/content/train.done"
REMOTE_PID = "/content/train.pid"


def run_colab(args: list[str], stdin_text: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = ["colab"] + args
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def exec_code(code: str, session: str, timeout: int = 300) -> tuple[int, str]:
    """在远端 kernel 里执行一段代码，返回 (exit_code, 输出)。"""
    p = run_colab(["exec", "-s", session, "--timeout", "60"], stdin_text=code, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def launch(session: str, envs: dict[str, str], run_name: str) -> int:
    # 1. 上传脚本
    print(f"上传 {SCRIPT.name} → {session}:{REMOTE_DIR}/")
    exec_code(f"import os; os.makedirs('{REMOTE_DIR}', exist_ok=True); print('ok')", session)
    p = run_colab(["upload", "-s", session, str(SCRIPT), f"{REMOTE_DIR}/train_colab.py"], timeout=300)
    if p.returncode != 0:
        print("上传失败：", (p.stderr or p.stdout)[-800:])
        return 1

    # 2. 用 nohup 起后台进程
    env_expr = " ".join(f"{k}={v}" for k, v in envs.items())
    code = f"""
import subprocess, os, shlex
env = os.environ.copy()
env.update({{ {", ".join(f'{k!r}: {v!r}' for k, v in envs.items())} }})
cmd = ['nohup', 'python', '-u', '{REMOTE_DIR}/train_colab.py']
with open('{REMOTE_LOG}', 'w') as logf:
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True, cwd='{REMOTE_DIR}')
open('{REMOTE_PID}', 'w').write(str(proc.pid))
try:
    os.remove('{REMOTE_DONE}')
except FileNotFoundError:
    pass
print('LAUNCHED pid=' + str(proc.pid))
"""
    rc, out = exec_code(code, session)
    print(out.strip()[-500:])
    if "LAUNCHED" not in out:
        print("启动失败")
        return 1

    print(f"\n已在远端后台启动，run_name={run_name}")
    print(f"  日志：{REMOTE_LOG}")
    print(f"  查看进度：python scripts/colab_launch.py --tail")
    print(f"  判断结束：python scripts/colab_launch.py --status")
    return 0


def tail(session: str, lines: int = 40) -> int:
    code = f"""
p = '{REMOTE_LOG}'
try:
    with open(p, encoding='utf-8', errors='replace') as f:
        ls = f.readlines()
    print(''.join(ls[-{lines}:]))
    print('--- 总行数', len(ls))
except FileNotFoundError:
    print('日志还没生成')
"""
    rc, out = exec_code(code, session)
    print(out)
    return rc


def status(session: str) -> int:
    code = f"""
import os, json
info = {{}}
for name, path in [('log', '{REMOTE_LOG}'), ('done', '{REMOTE_DONE}'), ('pid', '{REMOTE_PID}')]:
    info[name] = os.path.exists(path)
try:
    pid = int(open('{REMOTE_PID}').read().strip())
    info['pid'] = pid
    try:
        os.kill(pid, 0)
        info['alive'] = True
    except OSError:
        info['alive'] = False
except Exception as e:
    info['alive'] = None
    info['err'] = str(e)
if os.path.exists('{REMOTE_LOG}'):
    with open('{REMOTE_LOG}', encoding='utf-8', errors='replace') as f:
        ls = f.readlines()
    info['log_lines'] = len(ls)
    info['last'] = [l.rstrip() for l in ls[-6:]]
print('STATUS ' + json.dumps(info, ensure_ascii=False))
"""
    rc, out = exec_code(code, session)
    print(out)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="train")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--strong-aug", default="1")
    ap.add_argument("--tail", action="store_true", help="只打印日志尾部")
    ap.add_argument("--lines", type=int, default=40)
    ap.add_argument("--status", action="store_true", help="只看是否还在跑")
    args = ap.parse_args()

    if args.tail:
        return tail(args.session, args.lines)
    if args.status:
        return status(args.session)

    run_name = args.run_name or f"{args.model}_{args.img_size}_{args.epochs}ep"
    envs = {
        "COLAB_MODEL": args.model,
        "COLAB_EPOCHS": str(args.epochs),
        "COLAB_IMG_SIZE": str(args.img_size),
        "COLAB_BATCH_SIZE": str(args.batch_size),
        "COLAB_NUM_WORKERS": str(args.num_workers),
        "COLAB_LR": args.lr,
        "COLAB_STRONG_AUG": args.strong_aug,
        "COLAB_RUN_NAME": run_name,
        "COLAB_BENCH": "0",
        "COLAB_PREPARE_ONLY": "0",
    }
    return launch(args.session, envs, run_name)


if __name__ == "__main__":
    sys.exit(main())
