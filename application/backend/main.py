
# application/backend/main.py
# FastAPI 后端入口

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .schemas import PredictRequest, PredictResponse, BrandsResponse, ModelsResponse, StatsResponse
from .pipeline_loader import get_brand_list, get_model_list, get_stats
from .prediction_service import predict, validate_input


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载所有 Pipeline 对象"""
    # 触发所有单例加载
    from .pipeline_loader import (
        get_preprocessor, get_feature_engineer, get_feature_selector,
        get_feature_names, get_model, get_shap_explainer, get_stats,
        get_brand_list, get_model_list,
    )
    get_preprocessor()
    get_feature_engineer()
    get_feature_selector()
    get_feature_names()
    get_model()
    # SHAP explainer 延迟加载（太重，首次请求时再加载）
    get_brand_list()
    get_model_list()
    get_stats()
    print("[FastAPI] 所有 Pipeline 对象已加载")
    yield


app = FastAPI(
    title="二手车价格预测系统 API",
    description="基于 18.8 万条真实交易数据的 LightGBM 智能估价引擎",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 允许 Vue 前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/brands", response_model=BrandsResponse)
async def get_brands():
    """获取所有可用品牌列表"""
    brands = get_brand_list()
    return BrandsResponse(brands=brands)


@app.get("/api/models", response_model=ModelsResponse)
async def get_models(brand: str = Query(..., description="品牌名称")):
    """根据品牌获取车型列表"""
    all_models = get_model_list()
    models = all_models.get(brand, [])
    if not models:
        # 如果精确匹配失败，尝试模糊匹配
        for b, ms in all_models.items():
            if brand.lower() in b.lower() or b.lower() in brand.lower():
                models = ms
                break
    return ModelsResponse(models=models)


@app.get("/api/stats", response_model=StatsResponse)
async def get_market_stats():
    """获取市场统计数据"""
    s = get_stats()
    return StatsResponse(**s)


@app.post("/api/predict", response_model=PredictResponse)
async def predict_price(req: PredictRequest):
    """核心 API：输入车辆信息，返回预测价格 + SHAP 解释"""
    try:
        result = predict(req)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"模型文件缺失: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/api/validate")
async def validate_fields(req: PredictRequest):
    """单独校验接口（不执行预测）"""
    warnings = validate_input(req)
    return {"valid": len(warnings) == 0, "warnings": warnings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application.backend.main:app", host="0.0.0.0", port=8000, reload=True)
