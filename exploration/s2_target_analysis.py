# exploration/s2_target_analysis.py
# ===================================
# 论文 2.2 目标变量分析
#
# 分析 price 的分布特征: 原始 vs log1p 对数变换
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, FIGURES_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_target_analysis():
    """运行 2.2 目标变量分析"""
    print("=" * 60)
    print("  2.2 目标变量分析 — Target Variable Analysis")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    price = train_df['price']

    print(f"\n  价格统计:")
    print(f"    样本量: {len(price):,}")
    print(f"    均值:   ${price.mean():,.0f}")
    print(f"    中位数: ${price.median():,.0f}")
    print(f"    标准差: ${price.std():,.0f}")
    print(f"    最小值: ${price.min():,.0f}")
    print(f"    最大值: ${price.max():,.0f}")
    print(f"    偏度:   {price.skew():.2f}")
    print(f"    峰度:   {price.kurtosis():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(price, bins=50, density=True, alpha=0.7, color='#C00000', edgecolor='white')
    axes[0].set_title(f'原始价格分布 (偏度={price.skew():.2f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Price (USD)')
    axes[0].set_ylabel('Density')

    log_price = np.log1p(price)
    axes[1].hist(log_price, bins=50, density=True, alpha=0.7, color='#2F5496', edgecolor='white')
    axes[1].set_title(f'对数变换后分布 (偏度={log_price.skew():.2f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('log(1 + Price)')
    axes[1].set_ylabel('Density')

    fig.suptitle('图 2-2 目标变量 (price) 分布对比: 原始 vs 对数变换',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, 'price_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 已保存: {save_path}")


if __name__ == "__main__":
    run_target_analysis()
