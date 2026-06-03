# application/app.py
# ===================
# 系统设计与实现 — Streamlit 交互式估价界面
#
# 功能:
#   1. 输入车辆参数 (里程/年份/马力/排量/燃料/事故/品牌/产权)
#   2. 合理性校验 (年份≤当前, 里程≥0, 马力范围等)
#   3. 预测价格 + 价格区间分类
#   4. 自然语言预测解读 (为什么这个价格?)
#
# Usage: streamlit run application/app.py
import sys, os, pickle
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, PROCESSED_TRAIN_PATH

# 页面配置
st.set_page_config(page_title="二手车价格预测系统", page_icon="🚗", layout="centered")
st.title("🚗 二手车价格预测系统")
st.markdown("基于 18.8 万条真实交易数据训练的 LightGBM 智能估价引擎")

# ---- 加载模型 ----
@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'ridge_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    stats = {}
    for col in df.columns:
        if col != 'log_price':
            stats[col] = {'mean': df[col].mean(), 'std': df[col].std(),
                          'min': df[col].min(), 'max': df[col].max()}
    stats['price'] = {'mean': np.expm1(df['log_price']).mean()}
    return stats

model = load_model()
stats = load_stats()

# ---- 输入表单 ----
st.header("📋 车辆参数")
col1, col2 = st.columns(2)
with col1:
    milage = st.number_input("行驶里程 (英里)", min_value=100, max_value=500000,
                             value=50000, step=1000)
    model_year = st.number_input("出厂年份", min_value=1990, max_value=2026, value=2018, step=1)
    engine_hp = st.number_input("发动机马力 (HP)", min_value=50, max_value=2000, value=200, step=10)
    engine_liter = st.number_input("发动机排量 (L)", min_value=0.5, max_value=10.0,
                                   value=2.0, step=0.1, format="%.1f")
with col2:
    fuel_type = st.selectbox("燃油类型", ['Gasoline', 'Diesel', 'Electric',
                                           'Hybrid', 'E85 Flex Fuel', 'Plug-In Hybrid', 'Unknown'])
    accident_status = st.selectbox("事故记录", ['No_Accident', 'Has_Accident', 'Unknown'])
    brand = st.selectbox("品牌", ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes-Benz',
                                  'Audi', 'Volkswagen', 'Nissan', 'Hyundai', 'Tesla', 'Other'])
    is_clean_title = st.selectbox("产权状态 (Clean Title)", ['Yes', 'No'])

# ---- 预测 ----
if st.button("💰 预测价格", type="primary", use_container_width=True):
    current_year = 2026
    car_age = current_year - model_year
    errors = []
    if car_age < 0: errors.append("❌ 出厂年份不能超过当前年份")
    if car_age > 50: errors.append("⚠️ 车龄超过50年, 预测可能不准确")
    if milage < 0: errors.append("❌ 里程不能为负数")
    if engine_hp < 20 or engine_hp > 2000: errors.append("⚠️ 马力数值异常")

    if errors:
        for e in errors: st.error(e)
    else:
        # 构建特征
        annual_milage = milage / (car_age + 1)
        power_density = engine_hp / max(engine_liter, 0.1)
        feat = pd.DataFrame([{
            'milage': milage, 'engine_hp': engine_hp, 'car_age': car_age,
            'annual_milage': annual_milage, 'power_density': power_density,
            'brand_encoded': stats.get('brand_encoded', {'mean': 0}).get('mean', 0),
            'is_clean_title': 1 if is_clean_title == 'Yes' else 0,
            'fuel_type_Electric': 1 if fuel_type == 'Electric' else 0,
            'fuel_type_Hybrid': 1 if fuel_type == 'Hybrid' else 0,
            'accident_status_No_Accident': 1 if accident_status == 'No_Accident' else 0,
        }])

        pred_log = model.predict(feat.values)[0]
        pred_price = np.expm1(pred_log)

        # 结果展示
        st.header("📊 预测结果")
        if pred_price < 15000: tier = "低端代步车"
        elif pred_price < 35000: tier = "中端实用车"
        elif pred_price < 75000: tier = "高档轻奢车"
        else: tier = "顶级豪华超跑"

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("预测价格", f"${pred_price:,.0f}", delta=tier)
        with col_b:
            avg_price = stats['price']['mean']
            st.metric("市场均价", f"${avg_price:,.0f}",
                      delta=f"{'高于' if pred_price > avg_price else '低于'}均价")

        # 预测解读
        st.subheader("📝 预测解读")
        reasons = []
        if car_age > 8: reasons.append(f"🔻 车龄 {car_age} 年偏长, 折旧因素拉低价格")
        elif car_age < 3: reasons.append(f"🔺 车龄仅 {car_age} 年, 保值率较高")
        if milage > 100000: reasons.append(f"🔻 里程 {milage:,} 英里偏高, 磨损程度较大")
        elif milage < 30000: reasons.append(f"🔺 里程仅 {milage:,} 英里, 车况接近新车")
        if accident_status == 'Has_Accident': reasons.append("🔻 有事故记录, 影响车辆残值")
        if engine_hp > 300: reasons.append("🔺 发动机马力较高, 性能溢价")
        if power_density > 100: reasons.append("🔺 升功率较高, 反映发动机技术先进")
        if not reasons: reasons.append("✅ 该车辆各项参数处于正常区间")
        for r in reasons: st.write(r)
