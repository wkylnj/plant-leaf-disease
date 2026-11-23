"""Colab CLI 登录（一次性）。

背景与关键提示
--------------
Google 的授权页会为每个权限列一个勾选框，**默认不全勾**。如果只勾了前几个，
Google 只授予那几个基础权限（openid / userinfo.email / userinfo.profile），
缺少 `colaboratory` 就会导致后续所有 colab 命令 403。

所以授权时务必点「全选」或逐个勾上全部复选框。

本脚本相对 `colab usage` 的改进：
1. 设 OAUTHLIB_RELAX_TOKEN_SCOPE=1，避免 Google 少给权限时 oauthlib 直接抛
   ScopeHasChanged 导致 token 根本存不下来；
2. 把「实际授予」与「申请」的 scope 逐条对比打印，缺哪个一目了然；
3. 授权码失效（invalid_grant）时给出可读提示，而不是一屏 traceback。

用法：
    python scripts\\colab_login.py
"""

import json
import os
import sys
import time
from importlib import resources
from pathlib import Path

# 必须在 import oauthlib 之前设置
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

import keyring  # noqa: E402
from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
KEYRING_SERVICE_NAME = "google-colab-cli"
KEYRING_USERNAME = "oauth_token"

# 缺了这些就无法工作，单独强调
CRITICAL = {
    "https://www.googleapis.com/auth/colaboratory": "Colab 运行时服务必需",
    "https://www.googleapis.com/auth/userinfo.email": "会话后端必需",
}


def load_client_config() -> dict:
    try:
        res = resources.files("colab_cli").joinpath("oauth_config.json")
        if res.is_file():
            return json.loads(res.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] importlib.resources 失败: {e}", file=sys.stderr)

    import site

    for sp in site.getsitepackages():
        p = Path(sp) / "colab_cli" / "oauth_config.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    alt = Path.home() / ".colab-cli-oauth-config.json"
    if alt.exists():
        return json.loads(alt.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到 oauth_config.json")


def short(scope: str) -> str:
    return scope.rsplit("/auth/", 1)[-1]


def main() -> int:
    cfg = load_client_config()
    flow = InstalledAppFlow.from_client_config(cfg, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

    print("\n" + "=" * 72)
    print("第 1 步  复制下面这个链接到浏览器打开")
    print("=" * 72 + "\n")
    print(auth_url)
    print("\n" + "=" * 72)
    print("第 2 步  登录后会出现「选择要授予的权限」页面")
    print("        ★ 必须把每个复选框都勾上（点『全选』），否则会缺 colaboratory")
    print("          权限，后面所有 colab 命令都会 403。★")
    print("=" * 72)
    print("第 3 步  授权完成后浏览器显示一串授权码，复制它粘贴到下面")
    print("        （授权码只能用一次，且约 10 分钟内有效）")
    print("=" * 72 + "\n")

    # 授权码来源：
    #   1) --code 直接传（一次性，便于脚本化）
    #   2) --code-file 轮询文件（进程常驻等用户把码写进文件，适合后台跑）
    #   3) 交互输入（默认，人工在终端里粘贴）
    # 之所以要 2)：code_verifier 只存在本进程内存里，进程一退出链接就作废，
    # 所以必须让同一个进程活着等授权码，不能"打印链接→退出→下次再输入码"。
    code = ""
    if "--code" in sys.argv:
        code = sys.argv[sys.argv.index("--code") + 1].strip()
    elif "--code-file" in sys.argv:
        cf = Path(sys.argv[sys.argv.index("--code-file") + 1])
        print(f"等待授权码写入文件: {cf}")
        print("（最多等 15 分钟，把浏览器显示的授权码整行写进该文件即可）\n")
        deadline = time.time() + 15 * 60
        while time.time() < deadline:
            if cf.exists():
                code = cf.read_text(encoding="utf-8").strip()
                if code:
                    break
            time.sleep(2)
    else:
        code = input("授权码: ").strip()

    if not code:
        print("\n[错误] 没有拿到授权码（超时或为空）。", file=sys.stderr)
        return 1

    try:
        flow.fetch_token(code=code)
    except Exception as e:
        name = type(e).__name__
        print(f"\n[失败] 换取 token 出错：{name}", file=sys.stderr)
        if "InvalidGrant" in name or "invalid_grant" in str(e):
            print(
                "原因：授权码已失效（被用过或超时）。\n"
                "办法：重新运行本脚本，用新链接重新授权，不要复用旧授权码。",
                file=sys.stderr,
            )
        else:
            print(f"详情：{e}", file=sys.stderr)
        return 1

    creds = flow.credentials
    granted = list(creds.scopes or [])

    print("\n" + "-" * 72)
    print("实际授予的权限：")
    for s in granted:
        print("  [√] " + short(s))
    missing = [s for s in PUBLIC_SCOPES if s not in granted]
    if missing:
        print("\n没有授予的权限：")
        for s in missing:
            note = CRITICAL.get(s, "")
            print(f"  [×] {short(s)}" + (f"   ← {note}" if note else ""))

    # 无论权限是否齐全都先存下来，方便后续排查
    keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, creds.to_json())
    saved = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
    if not saved:
        print("\n[错误] token 写入 Windows 凭据管理器失败。", file=sys.stderr)
        return 1
    print("\ntoken 已存入 Windows 凭据管理器。")

    critical_missing = [s for s in CRITICAL if s not in granted]
    if critical_missing:
        print("\n" + "!" * 72)
        print("警告：缺少关键权限，colab 命令会失败。")
        print("请重新运行本脚本，在授权页面把全部复选框都勾上。")
        print("!" * 72)
        return 2

    print("\n权限齐全。下一步验证：colab sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
