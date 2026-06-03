# modeling/s2_hyperparameter_tuning.py
# =====================================
# 论文 4.2 超参数调优
#
# Ridge/RF: 手动网格搜索 (解释参数变化对模型的影响)
# LGB/XGB/Cat: Optuna 贝叶斯优化
#
# 对 ≥2 个关键参数绘制参数-性能曲线
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_hyperparameter_tuning(n_trials=30):
    """运行 4.2 超参数调优"""
    safe_print("=" * 60)
    safe_print("  4.2 超参数调优")
    safe_print(f"  Optuna trials: {n_trials}")
    safe_print("=" * 60)

    from src.tune_all_models import run_full_tuning
    run_full_tuning(n_trials=n_trials)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=30)
    args = p.parse_args()
    run_hyperparameter_tuning(args.trials)
