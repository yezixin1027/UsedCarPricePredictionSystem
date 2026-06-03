# preprocessing/feature_engineering.py
# ========================================
# 3.3 特征工程
#
# 调用 src/features.py 中的 HighScoreFeatureEngineer:
#   衍生特征(4): annual_milage, power_density, car_age_squared, hp_per_year
#   目标编码(2): brand_encoded, model_encoded (5折嵌套OOF + 拉普拉斯平滑)
#   One-Hot编码: fuel_type, accident_status
#   Z-score标准化: 全部连续数值特征
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_feature_engineering():
    """运行 3.3 特征工程演示"""
    safe_print("=" * 60)
    safe_print("  3.3 特征工程 — Feature Engineering")
    safe_print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['price'])
    y = train_df['price']

    preprocessor = AdvancedUsedCarPreprocessor()
    X_clean = preprocessor.fit_transform(X)
    engineer = HighScoreFeatureEngineer()
    X_feat = engineer.fit_transform(X_clean, y)

    safe_print(f"\n[特征工程结果]")
    safe_print(f"  输入维度: {X_clean.shape[1]} (清洗后)")
    safe_print(f"  输出维度: {X_feat.shape[1]} (特征工程后)")
    safe_print(f"  新增特征: {X_feat.shape[1] - X_clean.shape[1] + 2} 个 (brand/model转为编码列)")

    safe_print(f"\n[特征清单]")
    for col in X_feat.columns:
        safe_print(f"  {col}")

    safe_print(f"\n[设计思路]")
    safe_print("  1. annual_milage = milage/(car_age+1): 物理磨损强度, 消除车龄混淆")
    safe_print("  2. power_density = HP/Liter: 升功率, 衡量发动机技术水平")
    safe_print("  3. car_age_squared: 捕获折旧非线性(前3年加速折旧)")
    safe_print("  4. hp_per_year = HP/(car_age+1): 马力保有量, 衡量老化速度")
    safe_print("  (brand_x_milage, milage_log, engine_torque_proxy 经VIF+MI验证无效, 已移除)")
    safe_print(f"\n  [OK] 3.3 特征工程完成")
    return X_feat


if __name__ == "__main__":
    run_feature_engineering()
