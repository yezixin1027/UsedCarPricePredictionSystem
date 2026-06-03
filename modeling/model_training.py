# modeling/model_training.py
# =============================
# 4.1-4.2 模型选择与训练
#
# 5种模型: Ridge, Random Forest, LightGBM, XGBoost, CatBoost
# 统一使用 5折分层CV (按价格分位数stratify) 确保公平对比
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH, AVAILABLE_MODELS
from src.train import train_all_models, generate_comparison_table


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_model_training():
    """运行 4.1-4.2 模型训练"""
    safe_print("=" * 60)
    safe_print("  4.1-4.2 模型选择与训练")
    safe_print("=" * 60)

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    y = np.expm1(train_df['log_price'].values)

    safe_print(f"\n[数据] {X.shape[0]:,} 样本 x {X.shape[1]} 特征")

    # 算法对比表
    safe_print(f"\n[4.1] 算法核心假设与适用条件:")
    table = generate_comparison_table()
    for _, row in table.iterrows():
        safe_print(f"  {row['算法']}")
        safe_print(f"    类型: {row['模型类型']}")
        safe_print(f"    假设: {row['核心假设'][:80]}...")
        safe_print(f"    优势: {row['优势']}")

    # 5模型训练
    safe_print(f"\n[4.2] 五模型训练 (5-fold Stratified CV)")
    results = train_all_models(X, y, models=AVAILABLE_MODELS, cv_folds=5)

    safe_print(f"\n  {'Model':<18} {'R2':>8} {'+/-':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    safe_print(f"  {'─' * 72}")
    for _, row in results.iterrows():
        safe_print(f"  {row['model']:<18} {row['R2_mean']:>8.4f} {row['R2_std']:>8.4f} "
                   f"{row['MAE_mean']:>12,.0f} {row['RMSE_mean']:>12,.0f} {row['MAPE_mean']:>7.1f}%")

    best = results.iloc[0]
    safe_print(f"\n  [最优单模型] {best['model']} R2={best['R2_mean']:.4f} MAE={best['MAE_mean']:,.0f}")
    safe_print(f"\n  [OK] 4.1-4.2 模型训练完成")
    return results


if __name__ == "__main__":
    run_model_training()
