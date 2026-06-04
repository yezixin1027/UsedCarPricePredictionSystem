# application/backend/schemas.py
# Pydantic 请求/响应模型

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    brand: str = Field(..., description="品牌", examples=["Toyota"])
    model: str = Field(..., description="车型", examples=["Camry"])
    model_year: int = Field(..., ge=1990, le=2026, description="出厂年份", examples=[2020])
    milage: int = Field(..., ge=100, le=400000, description="行驶里程(英里)", examples=[45000])
    engine_hp: float = Field(..., ge=50, le=1500, description="发动机马力(HP)", examples=[203])
    engine_liter: float = Field(..., ge=0.5, le=8.0, description="发动机排量(L)", examples=[2.5])
    fuel_type: str = Field(..., description="燃油类型", examples=["Gasoline"])
    accident: str = Field(..., description="事故记录", examples=["None reported"])
    clean_title: str = Field(..., description="产权状态", examples=["Yes"])

    class Config:
        json_schema_extra = {
            "example": {
                "brand": "Toyota",
                "model": "Camry",
                "model_year": 2020,
                "milage": 45000,
                "engine_hp": 203,
                "engine_liter": 2.5,
                "fuel_type": "Gasoline",
                "accident": "None reported",
                "clean_title": "Yes",
            }
        }


class ShapItem(BaseModel):
    feature: str
    display_name: str
    contribution: float  # 美元贡献值
    direction: str       # "positive" or "negative"


class PredictResponse(BaseModel):
    predicted_price: float
    price_tier: str
    market_avg_price: float
    percentile: int
    shap_explanation: List[ShapItem]
    natural_language: List[str]
    validation_warnings: List[str] = []


class BrandsResponse(BaseModel):
    brands: List[str]


class ModelsResponse(BaseModel):
    models: List[str]


class StatsResponse(BaseModel):
    avg_price: float
    total_samples: int
    model_r2: float
    model_mae: float
