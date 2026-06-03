# exploration/s1_dataset_overview.py
# ======================================
# 论文 2.1 数据集概览
#
# 输出内容:
#   1. 数据集基本信息（样本量、特征数、内存占用）
#   2. 各列数据类型与缺失值统计
#   3. 数值特征描述性统计（均值/标准差/分位数/偏度/峰度）
#   4. 分类特征基数统计
#   5. 数据质量初步评估
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, TEST_PATH


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_dataset_overview():
    """运行 2.1 数据集概览分析"""
    safe_print("=" * 70)
    safe_print("  2.1 数据集概览 — Dataset Overview")
    safe_print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    safe_print(f"\n[1] 数据规模:")
    safe_print(f"  训练集: {train_df.shape[0]:,} 条记录 x {train_df.shape[1]} 个字段")
    safe_print(f"  测试集: {test_df.shape[0]:,} 条记录 x {test_df.shape[1]} 个字段")
    safe_print(f"  训练集内存: {train_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    safe_print(f"  测试集内存: {test_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    safe_print(f"\n[2] 字段信息与缺失值统计 (训练集):")
    safe_print(f"  {'字段名':<18} {'类型':<12} {'缺失数':>8} {'缺失率':>8}")
    safe_print(f"  {'─' * 48}")
    for col in train_df.columns:
        dtype = str(train_df[col].dtype)
        missing = train_df[col].isna().sum()
        missing_pct = missing / len(train_df) * 100
        safe_print(f"  {col:<18} {dtype:<12} {missing:>8} {missing_pct:>7.2f}%")

    safe_print(f"\n[3] 数值特征描述性统计 (训练集):")
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    desc = train_df[num_cols].describe().T
    desc['skewness'] = train_df[num_cols].skew()
    for col in num_cols:
        row = desc.loc[col]
        safe_print(f"  {col:<15} mean={row['mean']:>12,.1f}  std={row['std']:>12,.1f}  "
                   f"min={row['min']:>10,.1f}  max={row['max']:>12,.1f}  "
                   f"skew={row['skewness']:>7.2f}")

    safe_print(f"\n[4] 分类特征基数分析 (训练集):")
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        n_unique = train_df[col].nunique()
        top3 = train_df[col].value_counts().head(3).index.tolist()
        safe_print(f"  {col:<18} 唯一值: {n_unique:>6}  最常见: {top3}")

    safe_print(f"\n[5] 数据质量初步评估:")
    total_cells = train_df.shape[0] * train_df.shape[1]
    total_missing = train_df.isna().sum().sum()
    overall_missing_rate = total_missing / total_cells * 100
    safe_print(f"  总缺失率: {overall_missing_rate:.2f}%")

    if 'price' in train_df.columns:
        safe_print(f"  价格: min={train_df['price'].min():,.0f}, "
                   f"median={train_df['price'].median():,.0f}, "
                   f"max={train_df['price'].max():,.0f}")

    if 'milage' in train_df.columns:
        neg_milage = (train_df['milage'] < 0).sum()
        zero_milage = (train_df['milage'] == 0).sum()
        safe_print(f"  里程异常: 负数={neg_milage}, 零值={zero_milage}")

    safe_print(f"\n  [OK] 2.1 数据集概览 完成")
    return train_df


if __name__ == "__main__":
    run_dataset_overview()
