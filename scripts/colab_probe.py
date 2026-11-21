"""最小验证：colab run 是否可用。"""

import os
import subprocess
import sys

print("PYTHON", sys.version.split()[0])
print("CWD", os.getcwd())
try:
    import torch

    print("TORCH", torch.__version__)
    print("CUDA", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU", torch.cuda.get_device_name(0))
        print("VRAM_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
except Exception as e:
    print("TORCH_ERR", type(e).__name__, e)

print("NPROC", os.cpu_count())
for cmd in ["nvidia-smi --query-gpu=name --format=csv,noheader", "df -h /content | tail -1", "ls /content"]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("CMD:", cmd)
    print((p.stdout or p.stderr).strip()[:400])

print("HELLO_FROM_COLAB_RUN")
