"""Kaggle 冒烟测试：确认 GPU、联网、torch 可用。"""

import os
import subprocess
import sys
import urllib.request

def log(msg):
    print(f"[smoke] {msg}", flush=True)

log("=" * 60)
log(f"python {sys.version.split()[0]}")
log(f"cwd={os.getcwd()}")
log(f"nproc={os.cpu_count()}")

import torch
log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    log(f"gpu={p.name} vram={p.total_memory/1e9:.1f}GB")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    log(f"matmul ok, sum={y.sum().item():.2f}")

import torchvision
log(f"torchvision={torchvision.__version__}")

# 联网测试
try:
    with urllib.request.urlopen("https://huggingface.co", timeout=20) as r:
        log(f"huggingface 可达: HTTP {r.status}")
except Exception as e:
    log(f"huggingface 不可达: {e}")

# 磁盘
st = os.statvfs("/kaggle/working") if hasattr(os, "statvfs") else None
if st:
    log(f"/kaggle/working 可用 {st.f_bavail*st.f_frsize/1e9:.1f}GB")

log("SMOKE_OK")
