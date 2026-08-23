#!/usr/bin/env python
"""
后端启动脚本
从 backend 目录运行: python run.py            # 单进程常驻（推荐，稳定）
          可选:    python run.py --reload     # 热重载（仅开发时用，Windows 下不可靠）
"""
import sys
import os
from pathlib import Path

# 添加 backend 目录到 Python 路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn

    reload = "--reload" in sys.argv or os.environ.get("RELOAD") == "1"
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        # 只监听 src 源码目录（绝对路径），避免上传的数据集（可能含 .py 文件）
        # 解压进 data/ 时触发热重载导致后端重启
        reload_dirs=[str(backend_dir / "src")]
    )
