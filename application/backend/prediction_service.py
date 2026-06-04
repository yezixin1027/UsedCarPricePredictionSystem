# application/backend/prediction_service.py
# 预测推理服务：校验 → 变换 → 预测 → SHAP 解释

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .pipeline_loader import (
    get_preprocessor, get_feature_engineer, get_feature_selector,
    get_feature_names, get_model, get_shap_explainer, get_stats,
)
from .schemas import PredictRequest, PredictResponse, ShapItem

# SHAP 特征名 → 中文显示名映射
SHAP_DISPLAY_NAMES = {
    'hp_per_year': '年均马力保有量',
    'milage': '行驶里程',
    'car_age_squared': '车龄折旧效应',
    'brand_encoded': '品牌溢价',
    'power_density': '发动机升功率',
    'engine_hp': '发动机马力',
    'model_encoded': '车型保值率',
    'annual_milage': '年均行驶里程',
    'accident_status_No_Accident': '无事故记录',
    'fuel_type_Electric': '电动车型',
    'is_clean_title': '清晰产权',
}

PRICE_TIERS = [
    (0, 15000, '低端代步车'),
    (15000, 35000, '中端实用车'),
    (35000, 75000, '高档轻奢车'),
    (75000, float('inf'), '顶级豪华车'),
]


def validate_input(req: PredictRequest) -> list[str]:
    """业务校验，返回警告列表"""
    warnings = []
    current_year = 2026  # 数据集截止年份

    car_age = current_year - req.model_year
    if car_age < 0:
        warnings.append("出厂年份不能超过当前年份")
    if car_age > 40:
        warnings.append(f"车龄 {car_age} 年较长，预测仅供参考")
    if req.milage < 500:
        warnings.append("里程过低，请核实是否为真实里程")
    if req.milage > 300000:
        warnings.append(f"里程 {req.milage:,} 英里偏高，预测可能偏差较大")

    annual_milage = req.milage / max(car_age, 1)
    if car_age > 0 and annual_milage > 50000:
        warnings.append(f"年均里程 {annual_milage:,.0f} 英里异常偏高，请核实")

    if req.engine_hp < 60:
        warnings.append("马力数值偏低，请核实")
    if req.engine_liter < 0.8:
        warnings.append("排量数值偏低，请核实")

    return warnings


def predict(req: PredictRequest) -> PredictResponse:
    """完整的推理管线"""
    # 1. 校验
    warnings = validate_input(req)

    # 2. 构建原始 DataFrame（与训练数据格式一致）
    raw_input = pd.DataFrame([{
        'brand': req.brand,
        'model': req.model,
        'model_year': req.model_year,
        'milage': req.milage,
        'fuel_type': req.fuel_type,
        'engine': f"{req.engine_hp}HP {req.engine_liter}L Engine",
        'transmission': 'A/T',  # 默认值（特征工程后会丢弃）
        'ext_col': 'Black',      # 默认值
        'int_col': 'Black',      # 默认值
        'accident': req.accident,
        'clean_title': req.clean_title,
    }])

    # 3. Pipeline 变换
    preprocessor = get_preprocessor()
    fe_engineer = get_feature_engineer()
    selector = get_feature_selector()
    feature_names = get_feature_names()
    model = get_model()

    X_clean = preprocessor.transform(raw_input)
    X_feat = fe_engineer.transform(X_clean)

    # 确保所有训练特征都存在
    for col in feature_names:
        if col not in X_feat.columns:
            X_feat[col] = 0
    X_input = X_feat[feature_names].values.astype(float)

    # 4. 模型预测
    pred_log = model.predict(X_input)[0]
    pred_price = float(np.expm1(pred_log))
    pred_price = max(pred_price, 100)  # 价格下限

    # 5. 价格区间
    price_tier = None
    for lo, hi, tier in PRICE_TIERS:
        if lo <= pred_price < hi:
            price_tier = tier
            break

    # 6. 市场统计
    stats = get_stats()
    avg_price = stats['avg_price']
    # 估算分位数（简化：用线性插值）
    if pred_price <= avg_price:
        percentile = int(max(5, pred_price / avg_price * 50))
    else:
        percentile = int(min(95, 50 + (pred_price - avg_price) / avg_price * 50))

    # 7. SHAP 解释
    shap_items, nl_lines = _compute_shap_explanation(model, X_input, feature_names, pred_price, avg_price)

    return PredictResponse(
        predicted_price=pred_price,
        price_tier=price_tier,
        market_avg_price=round(avg_price, 0),
        percentile=percentile,
        shap_explanation=shap_items,
        natural_language=nl_lines,
        validation_warnings=warnings,
    )


def _compute_shap_explanation(model, X_input, feature_names, pred_price, avg_price):
    """计算 SHAP 值并生成自然语言解释"""
    try:
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # SHAP 值在 log 空间，转换为美元贡献
        base_log = explainer.expected_value
        if isinstance(base_log, list):
            base_log = base_log[0]
        base_price = np.expm1(base_log)

        # 对每个特征计算美元贡献
        contributions = []
        for i, fname in enumerate(feature_names):
            shap_log = shap_values[0][i]
            # 贡献 = exp(base_log + shap_log) - exp(base_log) ≈ shap_log * exp(base_log)
            contrib_usd = float(shap_log * np.exp(base_log))
            contributions.append({
                'feature': fname,
                'display_name': SHAP_DISPLAY_NAMES.get(fname, fname),
                'contribution': contrib_usd,
                'direction': 'positive' if contrib_usd > 0 else 'negative',
            })

        # 按绝对贡献排序
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # 取 Top-5
        top5 = contributions[:5]

        shap_items = [
            ShapItem(
                feature=c['feature'],
                display_name=c['display_name'],
                contribution=round(c['contribution'], 0),
                direction=c['direction'],
            )
            for c in top5
        ]

        # 生成自然语言
        nl_lines = []
        for c in top5:
            name = c['display_name']
            usd = abs(c['contribution'])
            if c['direction'] == 'positive':
                if usd > 1000:
                    nl_lines.append(f"[+] {name}：贡献 +${usd:,.0f}，显著提升预测价格")
                else:
                    nl_lines.append(f"[+] {name}：贡献 +${usd:,.0f}，对价格有正向支撑")
            else:
                if usd > 1000:
                    nl_lines.append(f"[-] {name}：拉低 -${usd:,.0f}，是价格下降的主要因素")
                else:
                    nl_lines.append(f"[-] {name}：拉低 -${usd:,.0f}，轻微压低价格")

        if not nl_lines:
            nl_lines.append("✅ 该车辆各项参数处于正常区间")

        return shap_items, nl_lines

    except Exception as e:
        # SHAP 计算失败时回退到简单解释
        return [], [f"预测价格 ${pred_price:,.0f}，市场均价 ${avg_price:,.0f}"]
