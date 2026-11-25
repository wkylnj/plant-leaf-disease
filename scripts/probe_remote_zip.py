"""探测远程 zip 里的图片数据集结构（不下载整包）。

用途：在正式下载前，用 HTTP Range 只读 zip 的中央目录，得到
「类别 → 图片数」的完整清单，从而判断数据集规模与类别分布是否
符合我们的跨域评估需要（例如是否覆盖 PlantVillage 的 38 类）。

用法：
    python scripts/probe_remote_zip.py --url <zip 直链> [--prefix <子目录>] [--top 20]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


class RangeFile(io.RawIOBase):
    """把远程文件伪装成可 seek 的本地文件，只在需要时按 Range 取块。"""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.pos = 0
        self.block = block
        self.buf = b""
        self.buf_start = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        in_buf = self.buf_start <= self.pos and end < self.buf_start + len(self.buf)
        if not in_buf:
            s = max(0, self.pos - 1024)
            e = min(self.size - 1, end + self.block)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.buf = r.read()
                self.buf_start = s
        o = self.pos - self.buf_start
        data = self.buf[o:o + n]
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def class_of(name: str, prefix: str) -> str | None:
    """从 zip 条目名推断类别：取 prefix 之后的第一个目录名。"""
    if prefix and not name.startswith(prefix):
        return None
    rest = name[len(prefix):] if prefix else name
    parts = [p for p in rest.split("/") if p]
    if len(parts) < 2:
        return None
    if not name.lower().endswith(IMG_EXT):
        return None
    return parts[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="探测远程 zip 的类别结构")
    ap.add_argument("--url", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rf = RangeFile(args.url)
    print(f"[probe] 远程文件 {rf.size / 1e6:.1f} MB")
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    names = zf.namelist()
    print(f"[probe] zip 条目 {len(names)}")

    counter = Counter()
    for n in names:
        c = class_of(n, args.prefix)
        if c:
            counter[c] += 1

    total = sum(counter.values())
    print(f"[probe] 图片 {total} 张，类别 {len(counter)} 个")
    sizes = sorted(counter.values())
    if sizes:
        print(f"[probe] 每类张数 min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}")
        print(f"[probe] 不均衡比 {sizes[-1] / max(sizes[0], 1):.1f}x")

    print(f"\n[probe] 类别清单（按张数降序，前 {args.top}）：")
    for c, k in counter.most_common(args.top):
        print(f"   {k:6d}  {c}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "url": args.url,
            "zip_bytes": rf.size,
            "entries": len(names),
            "images": total,
            "classes": len(counter),
            "per_class": dict(counter.most_common()),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[probe] 产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
