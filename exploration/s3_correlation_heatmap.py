# exploration/s3_correlation_heatmap.py
# =======================================
# 论文 2.3 特征相关性分析
#
# 计算核心数值特征与 log_price 的 Pearson 相关系数矩阵
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, FIGURES_DIR
from src.preprocess import AdvancedUsedCarPreprocessor

sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_correlation_analysis():
    """运行 2.3 特征相关性分析"""
    print("=" * 60)
    print("  2.3 特征相关性分析 — Correlation Analysis")
    print("=" * 60)

    train_raw = pd.read_csv(TRAIN_PATH)
    preprocessor = AdvancedUsedCarPreprocessor()
    y_log = np.log1p(train_raw['price']).reset_index(drop=True)
    X_clean = preprocessor.fit_transform(train_raw.drop(columns=['price']))

    X_clean = pd.DataFrame(X_clean).reset_index(drop=True)
    target_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']
    analysis_df = X_clean[target_cols].astype(float).copy()
    analysis_df['log_price'] = y_log
    analysis_df.columns = ['行驶里程', '发动机马力', '发动机排量', '车龄', '对数价格']

    corr_matrix = analysis_df.corr(method='pearson')
    print(f"\n  与对数价格的相关系数:")
    for col in ['行驶里程', '发动机马力', '发动机排量', '车龄']:
        print(f"    {col}: r = {corr_matrix.loc[col, '对数价格']:.4f}")

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                annot=True, fmt=".2f", linewidths=.5,
                cbar_kws={"shrink": .7, "label": "皮尔逊相关系数 (r)"}, square=True)
    plt.title("图 2-3 训练集核心连续特征与对数价格皮尔逊相关性矩阵热力图",
              fontsize=13, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "feature_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 已保存: {save_path}")


if __name__ == "__main__":
    run_correlation_analysis()
