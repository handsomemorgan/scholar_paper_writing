#!/usr/bin/env python3
"""
论文自动写作助手 - Web GUI 服务器

提供 REST API + SSE 实时进度推送 + 静态前端页面。
启动方式:
    python web_server.py              # 默认 http://127.0.0.1:5000
    python web_server.py --port 8080  # 自定义端口
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
import logging
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import (
    Flask, request, jsonify, send_from_directory,
    Response, stream_with_context,
)
from dotenv import load_dotenv

load_dotenv()

from orchestrator import PaperWritingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WebServer")

# ===================== Flask 应用 =====================
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

# 全局任务状态
active_tasks: dict = {}  # task_id -> {"status": ..., "progress": [], "result": ...}


# ===================== 前端页面 =====================
@app.route("/")
def index():
    """返回前端页面"""
    # 如果 templates/index.html 不存在，返回内嵌页面
    template_path = os.path.join(app.template_folder, "index.html")
    if os.path.exists(template_path):
        return send_from_directory(app.template_folder, "index.html")
    return _inline_index()


# ===================== API: 获取可选模型列表 =====================
@app.route("/api/models")
def get_models():
    """返回可用的 LLM 模型列表"""
    models = [
        {
            "id": "deepseek",
            "name": "DeepSeek",
            "model": "deepseek-chat",
            "description": "DeepSeek Chat API — 高性价比，中文友好",
            "available": bool(os.getenv("DEEPSEEK_API_KEY")),
        },
        {
            "id": "anthropic",
            "name": "Anthropic Claude",
            "model": "claude-sonnet-4-6",
            "description": "Claude 系列 — 长文写作能力卓越",
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
        },
        {
            "id": "openai",
            "name": "OpenAI GPT",
            "model": "gpt-4o",
            "description": "OpenAI GPT 系列 — 生态丰富",
            "available": bool(os.getenv("OPENAI_API_KEY")),
        },
    ]
    return jsonify({"models": models})


# ===================== API: 生成论文 (SSE 进度流) =====================
@app.route("/api/generate", methods=["POST"])
def generate_paper():
    """提交论文生成任务，SSE 流式返回进度"""
    data = request.get_json()
    if not data or "requirements" not in data:
        return jsonify({"error": "缺少 'requirements' 字段"}), 400

    requirements = data["requirements"].strip()
    if not requirements:
        return jsonify({"error": "论文要求不能为空"}), 400

    provider = data.get("provider", "deepseek")
    model = data.get("model", None)
    extra_instructions = data.get("extra_instructions", "")

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000) % 100000)

    def generate():
        """SSE 生成器"""
        progress_queue = queue.Queue()

        def progress_callback(phase, status, message, details=None):
            """将进度推入队列"""
            progress_queue.put({
                "type": "progress",
                "phase": phase,
                "status": status,
                "message": message,
                "details": details or {},
                "timestamp": time.time(),
            })

        # 在线程中运行 orchestrator
        result_holder = {"result": None, "error": None}

        def run_orchestrator():
            try:
                orchestrator = PaperWritingOrchestrator(
                    provider=provider,
                    model=model,
                )
                result = orchestrator.run(
                    requirements=requirements,
                    extra_instructions=extra_instructions,
                    progress_callback=progress_callback,
                )
                result_holder["result"] = result
            except Exception as e:
                logger.error(f"生成失败: {e}", exc_info=True)
                result_holder["error"] = str(e)
            finally:
                progress_queue.put({"type": "done", "timestamp": time.time()})

        thread = threading.Thread(target=run_orchestrator, daemon=True)
        thread.start()

        # 持续推送进度
        while thread.is_alive() or not progress_queue.empty():
            try:
                event = progress_queue.get(timeout=0.5)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event["type"] == "done":
                    break
            except queue.Empty:
                # 发送心跳保活
                yield f": heartbeat\n\n"

        thread.join()

        # 发送最终结果
        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': result_holder['error']}, ensure_ascii=False)}\n\n"
        elif result_holder["result"]:
            r = result_holder["result"]
            yield f"data: {json.dumps({'type': 'result', 'output_dir': r.get('output_dir'), 'paper_length': len(r.get('paper', '')), 'format_score': r['metadata'].get('format_score', 0), 'pipeline_log': {k: v for k, v in r.get('pipeline_log', {}).items() if k != 'classification'}}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ===================== API: 获取已生成的论文内容 =====================
@app.route("/api/paper/<path:subpath>")
def get_paper(subpath):
    """获取 output 目录下的论文文件"""
    output_path = os.path.join(os.path.dirname(__file__), subpath)
    if not os.path.exists(output_path):
        return jsonify({"error": "文件不存在"}), 404

    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    return send_from_directory(directory, filename)


# ===================== 内嵌前端（备选） =====================
def _inline_index():
    """如果 template 目录下的 index.html 不存则用这个"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>论文自动写作助手</title>
    <style>body{font-family:sans-serif;padding:40px;text-align:center}</style>
</head>
<body>
    <h1>论文自动写作助手</h1>
    <p>前端页面正在构建中，请使用以下 API：</p>
    <pre>POST /api/generate</pre>
    <pre>GET /api/models</pre>
</body>
</html>"""


# ===================== 启动入口 =====================
def main():
    parser = argparse.ArgumentParser(description="论文自动写作助手 - Web 服务器")
    parser.add_argument("--port", "-p", type=int, default=5000, help="服务端口 (默认: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定地址 (默认: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  论文自动写作助手 (Paper Writing Agent System)")
    print("  Web GUI 模式")
    print("=" * 60)
    print(f"\n  访问地址: http://{args.host}:{args.port}")
    print("  按 Ctrl+C 停止服务\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
