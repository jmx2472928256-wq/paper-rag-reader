# -*- coding: utf-8 -*-
"""
Web 前端启动入口：同时启动 FastAPI 后端 + Gradio 前端（同一进程、同一端口）。

用法
----
    python web.py                    # 启动并自动打开浏览器 http://127.0.0.1:7860/
    python web.py --port 8000        # 指定端口
    python web.py --no-browser       # 不自动打开浏览器

页面
----
    http://127.0.0.1:<port>/        Gradio 前端（文献库/摘要/问答）
    http://127.0.0.1:<port>/docs    FastAPI 接口文档（/api/papers 等）
"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

# 允许 `python web.py` 直接运行（无包上下文时把项目根加入 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import setup_console
from src.webapp import create_app


def main() -> None:
    setup_console()  # Windows 控制台 UTF-8 输出
    parser = argparse.ArgumentParser(description="学术文献 RAG 研读助手 Web 前端")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，仅本地）")
    parser.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    app = create_app()
    url = f"http://{args.host}:{args.port}/"

    if not args.no_browser:
        # 延迟打开浏览器，等 uvicorn 完成监听
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    print("=" * 70)
    print("学术文献 RAG 研读助手 · Web 前端")
    print(f"  页面:     {url}")
    print(f"  接口文档: http://{args.host}:{args.port}/docs")
    print("  按 Ctrl+C 退出")
    print("=" * 70)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
