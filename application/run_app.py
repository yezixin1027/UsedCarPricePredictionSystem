# application/run_app.py
# =====================
# 论文第6章 · 系统设计与实现 — 启动脚本
#
# 启动 Streamlit 交互式估价界面
# Usage: python application/run_app.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    subprocess.run(["streamlit", "run", app_path])
