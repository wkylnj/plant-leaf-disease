"""从 Kaggle kernel 的输出 zip 里只抽取指定文件（不整包下载）。

背景：`plant-leaf-fbr` 的输出目录混着整个 FBR 合成数据集（6648 个文件），
Kaggle 把它打成一个 5.3 GB 的 `_output_.zip`，`kaggle kernels output`
只能整包下载，本地断了三次都没拿回来。

但 Kaggle 的 API 会给这个 zip 一个**签名 URL**，支持 HTTP Range。
所以可以像读本地 zip 一样：只读中央目录，找到 `best.pt` 的位置，
再按 Range 取那一段字节。

两个坑：
1. 签名 URL 不响应 HEAD（404），只能 `Range: bytes=0-0` 再从
   `Content-Range: bytes 0-0/<total>` 里拿总长度。
2. urllib 直连会被服务器掐断（`RemoteDisconnected`），必须用 requests。

用法：
    python scripts/pull_kernel_file.py \
        --kernel wangkaiyu66/plant-leaf-fbr \
        --pattern "runs/.*fbr.*/(best\.pt|classes\.json|config\.json)$" \
        --out ml/runs/resnet18_fbr_224_15ep
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class RequestsRangeFile(io.RawIOBase):
    """把远程文件伪装成可 seek 的本地文件，只在需要时按 Range 取块。"""

    def __init__(self, url: str, block: int = 8 << 20, session=None):
        import requests

        self.url = url
        self.pos = 0
        self.block = block
        self.buf = b""
        self.buf_start = 0
        self.session = session or requests.Session()
        self.size = self._probe_size()

    def _probe_size(self) -> int:
        resp = self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True
        )
        resp.raise_for_status()
        cr = resp.headers.get("Content-Range", "")
        total = None
        if "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
        elif resp.headers.get("Content-Length"):
            total = int(resp.headers["Content-Length"])
        resp.close()
        if total is None:
            raise RuntimeError("无法取得远程文件大小")
        return total

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

    def _fetch(self, start: int, end: int) -> bytes:
        """带重试的 Range 请求。"""
        last = None
        for _ in range(4):
            try:
                resp = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.content
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"Range 请求连续失败：{type(last).__name__}: {last}")

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
            self.buf = self._fetch(s, e)
            self.buf_start = s
        o = self.pos - self.buf_start
        data = self.buf[o:o + n]
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def output_url(kernel: str) -> str:
    """拿 kernel 输出 zip 的签名下载 URL。"""
    from kaggle.api.kaggle_api_extended import (
        ApiListKernelSessionOutputRequest,
        KaggleApi,
    )

    owner, slug = kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.list_kernel_session_output(req)

    files = resp.files or []
    if not files:
        raise RuntimeError(f"{kernel} 没有输出文件")
    for it in files:
        if str(it.file_name).endswith(".zip"):
            return it.url
    return files[0].url


def extract(url: str, pattern: str, out_dir: Path, keep_structure: bool) -> list[Path]:
    rx = re.compile(pattern)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    rf = RequestsRangeFile(url)
    print(f"[pull] 远程 zip {rf.size / 1e6:.1f} MB", flush=True)
    with io.BufferedReader(rf, buffer_size=8 << 20) as buf:
        with zipfile.ZipFile(buf) as zf:
            names = [n for n in zf.namelist() if rx.search(n)]
            print(f"[pull] 命中 {len(names)} 个条目", flush=True)
            for name in names:
                info = zf.getinfo(name)
                dest = out_dir / name if keep_structure else out_dir / Path(name).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[pull] {name}  {info.file_size / 1e6:.2f} MB", flush=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                written.append(dest)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="从 kernel 输出 zip 里只取需要的文件")
    ap.add_argument("--kernel", required=True, help="owner/slug")
    ap.add_argument("--pattern", default="best.pt$", help="正则，匹配 zip 内路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-structure", action="store_true", help="保留 zip 内目录层级")
    ap.add_argument("--url", default="", help="直接给签名 URL（跳过 API 查询）")
    args = ap.parse_args()

    url = args.url or output_url(args.kernel)
    print(f"[pull] URL 前缀 {url[:70]}...", flush=True)
    written = extract(url, args.pattern, Path(args.out), args.keep_structure)
    if not written:
        print("[pull] 没有抽出任何文件", file=sys.stderr)
        return 1
    for p in written:
        print(f"[pull] 写入 {p}  {p.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
