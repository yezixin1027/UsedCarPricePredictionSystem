# application/run_frontend.py
# Start Vue frontend dev server

import os, sys, subprocess

if __name__ == "__main__":
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    os.chdir(frontend_dir)

    # Auto-install dependencies on first run
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("[首次运行] 正在安装前端依赖 (npm install)...")
        subprocess.run("npm install", shell=True, check=True)

    print("[启动] Vue 前端开发服务器 (http://localhost:5173)")
    subprocess.run("npm run dev", shell=True)
