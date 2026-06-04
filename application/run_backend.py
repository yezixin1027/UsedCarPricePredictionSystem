# application/run_backend.py
# 启动 FastAPI 后端
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application.backend.main:app", host="0.0.0.0", port=8000, reload=True)
