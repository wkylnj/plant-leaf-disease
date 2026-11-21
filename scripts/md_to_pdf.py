"""把 Markdown 转成适合上传博客的 PDF（借 Chrome 无头模式打印）。

用法：
    python scripts/md_to_pdf.py docs/周报/第11周周报.md
    python scripts/md_to_pdf.py docs/周报/第11周周报.md -o out.pdf

做了什么：
1. 用 python-markdown 把 md 转 HTML（开启 tables / fenced_code / toc 扩展）
2. 包一层打印友好的 CSS（中文字体、A4 页边距、表格边框、代码块底纹、图片自适应）
3. 调 Chrome --headless --print-to-pdf 输出

为什么不用 pandoc + xelatex：中文断行和图片浮动在 xelatex 下要额外配字体与模板，
Chrome 打印走的是网页渲染，中文、表格、图片的表现跟浏览器里看到的一致，最省事。
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt; line-height: 1.75; color: #1a1a1a; margin: 0;
}
h1 { font-size: 20pt; margin: 0 0 6mm 0; padding-bottom: 3mm;
     border-bottom: 2px solid #1F6F4A; color: #14472F; }
h2 { font-size: 14pt; margin: 8mm 0 3mm 0; padding-left: 3mm;
     border-left: 4px solid #1F6F4A; color: #14472F; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 6mm 0 2mm 0; color: #1F6F4A; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 4mm 0 2mm 0; color: #333; }
p { margin: 0 0 2.5mm 0; text-align: justify; }
ul, ol { margin: 0 0 2.5mm 0; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
strong { color: #14472F; font-weight: 600; }
blockquote {
  margin: 3mm 0; padding: 2mm 4mm; background: #F7F5EF;
  border-left: 3px solid #B45309; color: #555; font-size: 9.5pt;
}
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 9pt; background: #F2F2F2; padding: 0.5mm 1.2mm;
  border-radius: 2px; color: #B91C1C;
}
pre {
  background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 3px;
  padding: 3mm; overflow-x: auto; page-break-inside: avoid; margin: 3mm 0;
}
pre code { background: none; color: #1a1a1a; padding: 0; font-size: 8.5pt; line-height: 1.5; }
table {
  border-collapse: collapse; width: 100%; margin: 3mm 0;
  font-size: 9.5pt; page-break-inside: avoid;
}
th { background: #1F6F4A; color: #fff; font-weight: 600;
     padding: 2mm 2.5mm; text-align: left; border: 1px solid #1F6F4A; }
td { padding: 1.8mm 2.5mm; border: 1px solid #D8D8D8; vertical-align: top; }
tr:nth-child(even) td { background: #FAFAFA; }
img { max-width: 72%; height: auto; display: block; margin: 3mm auto;
      border: 1px solid #E0E0E0; border-radius: 2px; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #E0E0E0; margin: 6mm 0; }
a { color: #1F6F4A; text-decoration: none; }
"""


def find_browser() -> str:
    for p in CHROME_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("找不到 Chrome 或 Edge，无法打印 PDF")


def inline_images(html: str, base_dir: Path) -> str:
    """把本地图片转成 base64 内嵌。

    为什么必须内嵌：Chrome --headless 打印时用 file:// 打开的页面能读同目录图片，
    但相对路径跨目录（../figures/）容易因安全策略失败。内嵌最稳，代价是 HTML 变大。
    """
    pattern = re.compile(r'(<img[^>]+src=")([^"]+)(")')

    def repl(m: re.Match) -> str:
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (base_dir / src).resolve()
        if not path.exists():
            print(f"  [警告] 图片不存在，已跳过：{src}")
            return m.group(0)
        suffix = path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp", "svg": "svg+xml"}.get(suffix, "png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"{m.group(1)}data:image/{mime};base64,{b64}{m.group(3)}"

    return pattern.sub(repl, html)


def build_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"title": ""}},
    )
    body = inline_images(body, md_path.parent)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown → PDF（Chrome 无头打印）")
    ap.add_argument("src", help="Markdown 文件路径")
    ap.add_argument("-o", "--out", default="", help="输出 PDF 路径（默认与 md 同目录同名）")
    ap.add_argument("--keep-html", action="store_true", help="保留中间 HTML 便于排查样式")
    args = ap.parse_args()

    md_path = Path(args.src)
    if not md_path.is_absolute():
        md_path = (ROOT / md_path).resolve()
    if not md_path.exists():
        print(f"文件不存在：{md_path}")
        return 1

    out = Path(args.out) if args.out else md_path.with_suffix(".pdf")
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"转换 {md_path.name} …")
    html = build_html(md_path)

    if args.keep_html:
        html_path = out.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        print(f"  中间 HTML：{html_path}")

    with tempfile.TemporaryDirectory() as td:
        tmp_html = Path(td) / "page.html"
        tmp_html.write_text(html, encoding="utf-8")
        browser = find_browser()
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out}",
            tmp_html.as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0 and not out.exists():
            print("Chrome 打印失败：")
            print(proc.stderr[-2000:])
            return 1

    if not out.exists():
        print("PDF 未生成")
        return 1
    print(f"  已生成：{out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())