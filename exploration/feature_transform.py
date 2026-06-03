# exploration/feature_transform.py
# ====================================
# 2.4 数值特征变换策略对比分析
#
# Section A: 分布对比 — milage 与 engine_hp 在 4 种变换下的分布形态
# Section B: 模型影响 — Ridge 5折CV 对比 3 种变换策略的 R² 和 RMSE
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, FIGURES_DIR
from src.preprocess import AdvancedUsedCarPreprocessor

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def safe_skew(arr):
    clean = arr[np.isfinite(arr)]
    return stats.skew(clean) if len(clean) > 10 else np.nan


def run_feature_transform_analysis():
    """运行 2.4 特征变换策略对比分析"""
    print("=" * 60)
    print("  2.4 特征变换策略对比分析 — Feature Transform Analysis")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. 数据加载与预处理
    raw_df = pd.read_csv(TRAIN_PATH)
    y = raw_df['price']
    y_log = np.log1p(y)
    preprocessor = AdvancedUsedCarPreprocessor()
    X_clean = preprocessor.fit_transform(raw_df.drop(columns=['price']))
    raw_milage = X_clean['milage'].values.astype(float)
    raw_hp = X_clean['engine_hp'].values.astype(float)
    print(f"训练样本数: {len(raw_milage):,}")

    # ================================================================
    # Section A: 2x4 分布对比图
    # ================================================================
    print("\n[Section A] 生成 2x4 数值特征变换分布对比图")

    milage_log1p = np.log1p(raw_milage)
    scaler_m = StandardScaler()
    milage_zscore = scaler_m.fit_transform(raw_milage.reshape(-1, 1)).ravel()
    milage_log_zscore = scaler_m.fit_transform(milage_log1p.reshape(-1, 1)).ravel()

    hp_log1p = np.log1p(raw_hp)
    scaler_h = StandardScaler()
    hp_zscore = scaler_h.fit_transform(raw_hp.reshape(-1, 1)).ravel()
    hp_log_zscore = scaler_h.fit_transform(hp_log1p.reshape(-1, 1)).ravel()

    print(f"\n{'特征':<15} {'原始':>10} {'log1p':>10} {'Z-score':>10} {'log+Z':>10}")
    print("-" * 55)
    print(f"{'milage':<15} {safe_skew(raw_milage):>10.4f} {safe_skew(milage_log1p):>10.4f} "
          f"{safe_skew(milage_zscore):>10.4f} {safe_skew(milage_log_zscore):>10.4f}")
    print(f"{'engine_hp':<15} {safe_skew(raw_hp):>10.4f} {safe_skew(hp_log1p):>10.4f} "
          f"{safe_skew(hp_zscore):>10.4f} {safe_skew(hp_log_zscore):>10.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#7030A0']
    row_labels = ['行驶里程 (milage)', '发动机马力 (engine_hp)']
    col_labels = ['原始分布', 'log1p 对数变换', 'Z-score 标准化', 'log1p + Z-score\n(本项目采用)']
    data_rows = [
        [raw_milage, milage_log1p, milage_zscore, milage_log_zscore],
        [raw_hp, hp_log1p, hp_zscore, hp_log_zscore],
    ]

    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            arr = data_rows[i][j]
            clean = arr[np.isfinite(arr)]
            sk = safe_skew(arr)
            sns.histplot(clean, kde=True, bins=60, color=colors[j], alpha=0.6, ax=ax,
                         edgecolor='white', linewidth=0.3)
            ax.text(0.97, 0.92, f'偏度 = {sk:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                              edgecolor='gray', linewidth=0.5))
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=11, fontweight='bold')
            if i == 0:
                ax.set_title(col_labels[j], fontsize=11, fontweight='bold')
            if j >= 2:
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.tick_params(labelsize=8)

    fig.suptitle('图 2-4: 数值特征在不同变换策略下的分布形态对比', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_transform_distribution.png'), dpi=300)
    plt.close()
    print("  [OK] 分布对比图已保存")

    # ================================================================
    # Section B: 模型影响量化分析
    # ================================================================
    print("\n[Section B] 变换策略对 Ridge 回归性能的量化影响")

    num_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']
    X_num_raw = X_clean[num_cols].astype(float)
    annual_raw = (X_num_raw['milage'] / (X_num_raw['car_age'] + 1)).values.reshape(-1, 1)
    pd_raw_vals = (X_num_raw['engine_hp'] / X_num_raw['engine_liter'].replace(0, np.nan))
    pd_raw_vals = pd_raw_vals.replace([np.inf, -np.inf], np.nan).fillna(80).values.reshape(-1, 1)
    brand_mean = X_clean.groupby('brand').apply(
        lambda g: np.log1p(y[g.index]).mean(), include_groups=False)
    brand_encoded = X_clean['brand'].map(brand_mean).fillna(np.log1p(y).mean()).values.reshape(-1, 1)
    X_ohe = pd.get_dummies(X_clean[['fuel_type', 'accident_status']], drop_first=True, dtype=float)

    scaler_s1 = StandardScaler()
    X_num_clipped = np.clip(X_num_raw.values,
                            np.percentile(X_num_raw.values, 0.1, axis=0),
                            np.percentile(X_num_raw.values, 99.9, axis=0))
    X0 = np.hstack([X_num_clipped, annual_raw, pd_raw_vals, brand_encoded, X_ohe.values])

    X_num_z = scaler_s1.fit_transform(X_num_raw)
    annual_z = scaler_s1.fit_transform(annual_raw)
    pd_z = scaler_s1.fit_transform(pd_raw_vals)
    X1 = np.hstack([X_num_z, annual_z, pd_z, brand_encoded, X_ohe.values])

    scaler_r = RobustScaler()
    X_num_r = scaler_r.fit_transform(X_num_raw)
    annual_r = scaler_r.fit_transform(annual_raw)
    pd_r = scaler_r.fit_transform(pd_raw_vals)
    X2 = np.hstack([X_num_r, annual_r, pd_r, brand_encoded, X_ohe.values])

    ridge = Ridge(alpha=1.0, random_state=42)
    strategy_labels = ['原始特征\n(无标准化)', 'Z-score\n标准化(本项目)', 'RobustScaler\n(对比方法)']
    results = {}

    for name, X_data in zip(strategy_labels, [X0, X1, X2]):
        r2_scores = cross_val_score(ridge, X_data, y_log, cv=5, scoring='r2', n_jobs=-1)
        neg_mse = cross_val_score(ridge, X_data, y_log, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        results[name] = {'r2_mean': r2_scores.mean(), 'r2_std': r2_scores.std(),
                         'rmse_mean': np.sqrt(-neg_mse).mean(), 'rmse_std': np.sqrt(-neg_mse).std()}
        print(f"\n{name}")
        print(f"  R^2 = {results[name]['r2_mean']:.4f} +/- {results[name]['r2_std']:.4f}")
        print(f"  RMSE = {results[name]['rmse_mean']:.4f} +/- {results[name]['rmse_std']:.4f}")

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    names = list(results.keys())
    bar_colors = ['#C00000', '#2F5496', '#70AD47']

    r2_vals = [results[n]['r2_mean'] for n in names]
    r2_errs = [results[n]['r2_std'] for n in names]
    bars1 = ax1.bar(names, r2_vals, yerr=r2_errs, color=bar_colors, capsize=8, edgecolor='white', linewidth=1.2)
    ax1.set_title('5-fold CV R^2 (higher is better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R^2 Score', fontsize=11)
    for bar, val, err in zip(bars1, r2_vals, r2_errs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + max(r2_vals) * 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    rmse_vals = [results[n]['rmse_mean'] for n in names]
    rmse_errs = [results[n]['rmse_std'] for n in names]
    bars2 = ax2.bar(names, rmse_vals, yerr=rmse_errs, color=bar_colors, capsize=8, edgecolor='white', linewidth=1.2)
    ax2.set_title('5-fold CV RMSE (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (log space)', fontsize=11)
    for bar, val, err in zip(bars2, rmse_vals, rmse_errs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + max(rmse_vals) * 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    best_r2 = max(r2_vals)
    best_name = names[r2_vals.index(best_r2)]
    fig2.suptitle(f'图 2-4B: 特征变换策略对 Ridge Regression 性能影响\n最优: {best_name.strip()}',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_transform_model_impact.png'), dpi=300)
    plt.close()
    print("\n  [OK] 模型影响对比图已保存")
    print("\n[结论] Z-score 标准化为本项目采用方案")


if __name__ == "__main__":
    run_feature_transform_analysis()
