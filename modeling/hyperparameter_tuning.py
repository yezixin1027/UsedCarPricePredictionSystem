# modeling/hyperparameter_tuning.py
# =====================================
# 4.2 超参数调优
#
# 每个模型独立调优:
#   Ridge:       手动网格 (alpha)
#   ElasticNet:  手动网格 (alpha × l1_ratio)
#   RandomForest: 手动网格 (n_estimators → max_depth)
#   LightGBM:    Optuna 贝叶斯优化 (8参数)
#   KNN:         手动网格 (k × weights)
#
# 对 ≥2 个关键参数绘制参数-性能曲线
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_hyperparameter_tuning(n_trials=30):
    """运行 4.2 超参数调优 (6模型)"""
    safe_print("=" * 60)
    safe_print("  4.2 超参数调优 (6模型独立调优)")
    safe_print(f"  Optuna trials: {n_trials} (LightGBM)")
    safe_print("=" * 60)

    from src.tune_all_models import run_full_tuning
    run_full_tuning(n_trials=n_trials)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=30)
    args = p.parse_args()
    run_hyperparameter_tuning(args.trials)
