# preprocessing/feature_selection.py
# =======================================
#  3.5 特征选择与降维
#
# 两阶段筛选:
#   第一轮 VIF (方差膨胀因子): 剔除 VIF > 10 的共线性特征
#   第二轮 MI (互信息): 剔除 MI < 自适应阈值的弱相关特征
#
# 筛选标准:
#   VIF: VIF>10 → OLS系数方差膨胀10倍 → 模型解释性下降 → 剔除
#   MI:  互信息过低 → 与目标统计依赖极弱 → 对预测无贡献 → 剔除
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer
from src.selection import FeatureSelector


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_feature_selection():
    """运行 3.5 特征选择演示"""
    safe_print("=" * 60)
    safe_print("  3.5 特征选择与降维 — Feature Selection")
    safe_print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['price'])
    y = train_df['price']

    preprocessor = AdvancedUsedCarPreprocessor()
    X_clean = preprocessor.fit_transform(X)
    engineer = HighScoreFeatureEngineer()
    X_feat = engineer.fit_transform(X_clean, y)

    selector = FeatureSelector(vif_threshold=10.0, mi_percentile_factor=0.05, verbose=True)
    X_selected = selector.fit_transform(X_feat, y)

    safe_print(f"\n[筛选结果]")
    safe_print(f"  筛选前: {X_feat.shape[1]} 维")
    safe_print(f"  筛选后: {X_selected.shape[1]} 维")
    safe_print(f"  剔除:   {X_feat.shape[1] - X_selected.shape[1]} 个特征")
    safe_print(f"  降维率: {(1 - X_selected.shape[1]/X_feat.shape[1])*100:.1f}%")
    safe_print(f"  保留特征: {list(X_selected.columns)}")

    report = selector.get_report()
    if len(report) > 0:
        safe_print(f"\n[剔除详情]")
        for _, row in report.iterrows():
            safe_print(f"  [{row['round']}] {row['feature']}: {row['reason'][:80]}...")

    safe_print(f"\n[筛选标准说明]")
    safe_print("  VIF>10: 标准统计阈值, 协方差矩阵不稳定, 系数方差膨胀10倍")
    safe_print("  MI阈值: 所有特征MI均值的5%, 低于此值的特征预测贡献可忽略")
    safe_print(f"\n  [OK] 3.5 特征选择完成")
    return X_selected


if __name__ == "__main__":
    run_feature_selection()
