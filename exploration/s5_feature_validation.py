# exploration/s5_feature_validation.py
# =======================================
# 论文 2.5 衍生特征有效性综合验证
#
# Section A: 排列重要性 (Permutation Importance)
# Section B: Bootstrap 相关性置信区间 (B=1000)
# Section C: 偏依赖图 (Partial Dependence Plots)
# Section D: 消融实验 (Ablation Study)
# Section E: 综合验证仪表盘 (2x3 多面板汇总)
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, FIGURES_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def _permutation_importance(model, X, y, feature_names, n_repeats=10, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    importances = {name: [] for name in feature_names}
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        base_score = model_clone.score(X_val, y_val)
        for j, name in enumerate(feature_names):
            scores = []
            for _ in range(n_repeats):
                X_val_permuted = X_val.copy()
                np.random.shuffle(X_val_permuted[:, j])
                scores.append(base_score - model_clone.score(X_val_permuted, y_val))
            importances[name].extend(scores)
    results = []
    for name in feature_names:
        vals = np.array(importances[name])
        results.append({'feature': name, 'importance_mean': np.mean(vals), 'importance_std': np.std(vals)})
    return pd.DataFrame(results).sort_values('importance_mean', ascending=False)


def run_feature_validation():
    """运行 2.5 衍生特征有效性综合验证"""
    print("=" * 60)
    print("  2.5 衍生特征有效性综合验证 — Feature Validation")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. 数据加载与特征构建
    raw_df = pd.read_csv(TRAIN_PATH)
    y = raw_df['price']
    y_log = np.log1p(y)
    preprocessor = AdvancedUsedCarPreprocessor()
    fe_engineer = HighScoreFeatureEngineer()
    X_clean = preprocessor.fit_transform(raw_df.drop(columns=['price']))
    X_features = fe_engineer.fit_transform(X_clean, y)
    feature_names = X_features.columns.tolist()
    X_matrix = X_features.values.astype(float)
    n_samples, n_features = X_matrix.shape
    derived_features = ['annual_milage', 'power_density', 'brand_encoded']
    base_features = ['milage', 'engine_hp', 'engine_liter', 'car_age']

    print(f"样本: {n_samples:,} | 特征: {n_features}")

    # ================================================================
    # Section A: 排列重要性
    # ================================================================
    print("\n[Section A] 排列重要性分析")
    alphas = np.logspace(-2, 3, 20)
    ridge_cv = RidgeCV(alphas=alphas)
    pi_df = _permutation_importance(ridge_cv, X_matrix, y_log.values, feature_names, n_repeats=10, cv=5)

    safe_print(f"\n{'特征':<25} {'重要性':>12} {'标准差':>10}")
    safe_print("-" * 50)
    for _, row in pi_df.iterrows():
        safe_print(f"{row['feature']:<25} {row['importance_mean']:>12.6f} {row['importance_std']:>10.6f}")

    fig_a, ax = plt.subplots(figsize=(10, max(5, n_features * 0.45)))
    colors_a = ['#2F5496' if v > 0 else '#C00000' for v in pi_df['importance_mean']]
    ax.barh(range(len(pi_df)), pi_df['importance_mean'].values, xerr=pi_df['importance_std'].values,
            color=colors_a, edgecolor='white', linewidth=1.2, capsize=4, height=0.6)
    ax.set_yticks(range(len(pi_df)))
    ax.set_yticklabels(pi_df['feature'].values, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('R^2 decrease', fontsize=11)
    ax.set_title('Section A: Permutation Importance (10 repeats x 5-fold CV)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'validation_permutation_importance.png'), dpi=300)
    plt.close()
    safe_print("  [OK] 排列重要性图已保存")

    # ================================================================
    # Section B: Bootstrap 相关性置信区间
    # ================================================================
    print("\n[Section B] Bootstrap 相关性置信区间 (B=1000)")
    all_validate = derived_features + [f for f in base_features if f in feature_names]
    B = 1000
    bootstrap_results = {}
    rng = np.random.RandomState(42)

    for feat in all_validate:
        if feat not in X_features.columns:
            continue
        x_vals = X_features[feat].values
        pearson_vals, spearman_vals = [], []
        for _ in range(B):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            try:
                pr, _ = stats.pearsonr(x_vals[idx], y_log.values[idx])
                sr, _ = stats.spearmanr(x_vals[idx], y_log.values[idx])
                pearson_vals.append(pr); spearman_vals.append(sr)
            except Exception:
                continue
        pearson_vals = np.array(pearson_vals); spearman_vals = np.array(spearman_vals)
        bootstrap_results[feat] = {
            'pearson_mean': np.mean(pearson_vals),
            'pearson_ci_low': np.percentile(pearson_vals, 2.5),
            'pearson_ci_high': np.percentile(pearson_vals, 97.5),
            'spearman_mean': np.mean(spearman_vals),
            'spearman_ci_low': np.percentile(spearman_vals, 2.5),
            'spearman_ci_high': np.percentile(spearman_vals, 97.5),
        }

    safe_print(f"\n{'特征':<20} {'Pearson r':>10} {'95% CI':>22} {'Spearman rho':>13} {'95% CI':>22}")
    safe_print("-" * 90)
    for feat, res in bootstrap_results.items():
        safe_print(f"{feat:<20} {res['pearson_mean']:>10.4f} "
                   f"[{res['pearson_ci_low']:>7.4f}, {res['pearson_ci_high']:>7.4f}]"
                   f"  {res['spearman_mean']:>10.4f} "
                   f"[{res['spearman_ci_low']:>7.4f}, {res['spearman_ci_high']:>7.4f}]")

    fig_b, axes_b = plt.subplots(1, 2, figsize=(16, max(5, len(all_validate) * 0.45)))
    feat_list = list(bootstrap_results.keys())
    for ax_idx, (metric, title) in enumerate([('pearson', "Pearson r (95% CI)"), ('spearman', "Spearman rho (95% CI)")]):
        ax = axes_b[ax_idx]
        means = [bootstrap_results[f][f'{metric}_mean'] for f in feat_list]
        lows = [bootstrap_results[f][f'{metric}_ci_low'] for f in feat_list]
        highs = [bootstrap_results[f][f'{metric}_ci_high'] for f in feat_list]
        colors_b = ['#ED7D31' if f in derived_features else '#4472C4' for f in feat_list]
        for x, y_pos, xlo, xhi, mc in zip(means, range(len(feat_list)),
                                            [m - l for m, l in zip(means, lows)],
                                            [h - m for m, h in zip(means, highs)], colors_b):
            ax.errorbar(x, y_pos, xerr=[[xlo], [xhi]], fmt='o', capsize=5, elinewidth=1.5,
                        color='#333333', markersize=8, markerfacecolor=mc, markeredgecolor='white', markeredgewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_yticks(range(len(feat_list)))
        ax.set_yticklabels(feat_list, fontsize=10)
        ax.set_xlabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
    fig_b.suptitle('Section B: Bootstrap Correlation Confidence Intervals (B=1,000)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'validation_bootstrap_correlation.png'), dpi=300)
    plt.close()
    safe_print("  [OK] Bootstrap 相关性图已保存")

    # ================================================================
    # Section C: 偏依赖图
    # ================================================================
    print("\n[Section C] 偏依赖图")
    ridge_pdp = RidgeCV(alphas=alphas)
    ridge_pdp.fit(X_matrix, y_log.values)
    pdp_features = [f for f in all_validate if f in feature_names]
    pdp_indices = [feature_names.index(f) for f in pdp_features]
    n_pdp = len(pdp_features)
    n_cols = min(3, n_pdp)
    n_rows = int(np.ceil(n_pdp / n_cols))
    fig_c, axes_c = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_c = np.atleast_1d(axes_c).flatten()

    for i, (feat, feat_idx) in enumerate(zip(pdp_features, pdp_indices)):
        PartialDependenceDisplay.from_estimator(
            ridge_pdp, X_matrix, features=[feat_idx], feature_names=feature_names,
            kind='average', ax=axes_c[i], grid_resolution=50,
            line_kw={'color': '#2F5496', 'linewidth': 2})
        label = '(衍生)' if feat in derived_features else '(基础)'
        axes_c[i].set_title(f'{feat} {label}', fontsize=11, fontweight='bold')
    for i in range(n_pdp, len(axes_c)):
        axes_c[i].set_visible(False)
    fig_c.suptitle('Section C: Partial Dependence Plots (RidgeCV)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'validation_partial_dependence.png'), dpi=300)
    plt.close()
    safe_print("  [OK] 偏依赖图已保存")

    # ================================================================
    # Section D: 消融实验
    # ================================================================
    print("\n[Section D] 消融实验")
    base_scores = cross_val_score(ridge_cv, X_matrix, y_log.values, cv=5, scoring='r2', n_jobs=-1)
    safe_print(f"\n全特征基准 R^2 = {base_scores.mean():.4f} +/- {base_scores.std():.4f}")

    ablation_results = {}
    for drop_feat in derived_features:
        if drop_feat not in feature_names:
            continue
        keep_idx = [i for i, name in enumerate(feature_names) if name != drop_feat]
        X_ablated = X_matrix[:, keep_idx]
        ab_scores = cross_val_score(ridge_cv, X_ablated, y_log.values, cv=5, scoring='r2', n_jobs=-1)
        delta = base_scores.mean() - ab_scores.mean()
        ablation_results[drop_feat] = {'delta_r2': delta, 'delta_pct': delta / base_scores.mean() * 100}
        safe_print(f"剔除 {drop_feat:<20}: R^2 = {ab_scores.mean():.4f} | delta_R^2 = {delta:+.6f}")

    fig_d, ax_d = plt.subplots(figsize=(9, max(4, len(ablation_results) * 0.6)))
    feat_d = list(ablation_results.keys())
    deltas = [ablation_results[f]['delta_r2'] for f in feat_d]
    colors_d = ['#2F5496' if d > 0 else '#C00000' for d in deltas]
    bars = ax_d.barh(feat_d, deltas, color=colors_d, edgecolor='white', linewidth=1.2, height=0.5)
    ax_d.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width() + 0.0002 if delta >= 0 else bar.get_width() - 0.0008
        ax_d.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{delta:+.6f}', va='center', fontsize=11, fontweight='bold')
    ax_d.set_xlabel('delta R^2', fontsize=11)
    ax_d.set_title('Section D: Ablation Study — Feature Incremental Contribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'validation_ablation_study.png'), dpi=300)
    plt.close()
    safe_print("  [OK] 消融实验图已保存")

    # ================================================================
    # Section E: 综合验证仪表盘
    # ================================================================
    print("\n[Section E] 综合验证仪表盘")
    fig_e = plt.figure(figsize=(20, 12))

    ax1 = fig_e.add_subplot(2, 3, 1)
    colors_a2 = ['#2F5496' if v > 0 else '#C00000' for v in pi_df['importance_mean']]
    ax1.barh(range(len(pi_df)), pi_df['importance_mean'].values, xerr=pi_df['importance_std'].values,
             color=colors_a2, edgecolor='white', linewidth=1, capsize=3, height=0.6)
    ax1.set_yticks(range(len(pi_df))); ax1.set_yticklabels(pi_df['feature'].values, fontsize=9)
    ax1.invert_yaxis(); ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('R^2 decrease', fontsize=9)
    ax1.set_title('A: Permutation Importance', fontsize=11, fontweight='bold')

    ax2 = fig_e.add_subplot(2, 3, 2)
    means_p = [bootstrap_results[f]['pearson_mean'] for f in feat_list]
    lows_p = [bootstrap_results[f]['pearson_ci_low'] for f in feat_list]
    highs_p = [bootstrap_results[f]['pearson_ci_high'] for f in feat_list]
    colors_p = ['#ED7D31' if f in derived_features else '#4472C4' for f in feat_list]
    for x, y_pos, xlo, xhi, mc in zip(means_p, range(len(feat_list)),
                                        [m - l for m, l in zip(means_p, lows_p)],
                                        [h - m for m, h in zip(means_p, highs_p)], colors_p):
        ax2.errorbar(x, y_pos, xerr=[[xlo], [xhi]], fmt='o', capsize=4, elinewidth=1.2,
                     color='#333333', markersize=7, markerfacecolor=mc, markeredgecolor='white', markeredgewidth=0.8)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_yticks(range(len(feat_list))); ax2.set_yticklabels(feat_list, fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlabel('Pearson r', fontsize=9)
    ax2.set_title('B: Bootstrap Pearson r (95% CI)', fontsize=11, fontweight='bold')

    ax3 = fig_e.add_subplot(2, 3, 3)
    feat_d3 = list(ablation_results.keys())
    deltas3 = [ablation_results[f]['delta_r2'] for f in feat_d3]
    colors_d3 = ['#2F5496' if d > 0 else '#C00000' for d in deltas3]
    bars3 = ax3.barh(feat_d3, deltas3, color=colors_d3, edgecolor='white', linewidth=1, height=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    for bar, delta in zip(bars3, deltas3):
        x_pos = bar.get_width() + 0.0002 if delta >= 0 else bar.get_width() - 0.001
        ax3.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{delta:+.5f}', va='center', fontsize=10, fontweight='bold')
    ax3.set_xlabel('delta R^2', fontsize=9)
    ax3.set_title('D: Ablation Study', fontsize=11, fontweight='bold')

    pdp_derived_only = [f for f in derived_features if f in feature_names]
    for i, feat in enumerate(pdp_derived_only):
        ax_pdp = fig_e.add_subplot(2, 3, 4 + i)
        feat_idx = feature_names.index(feat)
        PartialDependenceDisplay.from_estimator(
            ridge_pdp, X_matrix, features=[feat_idx], feature_names=feature_names,
            kind='average', ax=ax_pdp, grid_resolution=40,
            line_kw={'color': '#ED7D31' if feat in derived_features else '#4472C4', 'linewidth': 2})
        ax_pdp.set_title(f'PDP: {feat}', fontsize=10, fontweight='bold')
        ax_pdp.set_xlabel(feat, fontsize=8); ax_pdp.set_ylabel('Partial Dependence', fontsize=8)

    fig_e.suptitle('Section E: Derived Feature Validation Dashboard', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'validation_dashboard.png'), dpi=300)
    plt.close()
    safe_print("  [OK] 综合验证仪表盘已保存")

    print("\n" + "=" * 60)
    print("  2.5 衍生特征有效性综合验证 完成")
    print(f"  输出目录: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_feature_validation()
