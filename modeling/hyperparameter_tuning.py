# modeling/hyperparameter_tuning.py
# =====================================
# 4.2 超参数调优
#
# 每个模型独立调优 + 参数-性能曲线:
#   Ridge:       手动网格 (alpha)           → tune_ridge_alpha.png
#   ElasticNet:  手动网格 (alpha × l1_ratio) → tune_elasticnet_heatmap.png
#   RandomForest: 手动网格 (n_est→max_depth) → tune_random_forest.png
#   LightGBM:    手动探索 + Optuna          → tune_lightgbm_manual.png
#   KNN:         手动网格 (k × weights)      → tune_knn.png
#
# 对 ≥2 个关键参数绘制参数-性能曲线
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_hyperparameter_tuning(n_trials=30):
    """运行 4.2 超参数调优 (5模型)"""
    safe_print("=" * 60)
    safe_print("  4.2 超参数调优 (5模型独立调优)")
    safe_print(f"  Optuna trials: {n_trials} (LightGBM)")
    safe_print("=" * 60)

    from src.tune_all_models import run_full_tuning
    best_params, final_results, stacking_result = run_full_tuning(n_trials=n_trials)

    # 调优完成后，用最佳模型绘制预测 vs 真实图
    safe_print(f"\n{'=' * 60}")
    safe_print("  预测 vs 真实可视化 (调优后最佳模型)")
    safe_print(f"{'=' * 60}")
    best_model = max(final_results, key=lambda k: final_results[k]['R2'])
    safe_print(f"  最佳模型: {best_model} (R²={final_results[best_model]['R2']:.4f})")

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    y = np.expm1(train_df['log_price'].values)

    from src.train import plot_predictions_vs_actual
    plot_predictions_vs_actual(best_model, X, y)
    safe_print(f"\n  [OK] 4.2 超参数调优完成")

    return best_params, final_results, stacking_result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=30)
    args = p.parse_args()
    run_hyperparameter_tuning(args.trials)
