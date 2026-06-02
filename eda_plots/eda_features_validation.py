# eda_plots/eda_features_validation.py
"""
衍生特征有效性综合验证
=======================

本脚本对特征工程产出的三个核心衍生特征进行多维度量化验证，
替代原有的单一 Pearson 散点图方案：

Section A · 排列重要性 (Permutation Importance)
    基于 RidgeCV 交叉验证模型，重复 10 次打乱每个特征列，量化
    特征被破坏后模型 R² 的平均下降幅度 ± 标准差。
    原理: 重要特征被打乱 → 预测性能显著下降；无关特征被打乱 → 性能不变。

Section B · Bootstrap 相关性置信区间
    从训练集中自助采样 (B=1000)，计算每个衍生特征与 log_price
    的 Pearson r 和 Spearman ρ，输出 95% 置信区间。
    相比单点估计，区间估计更稳健地反映相关性的统计显著性。

Section C · 偏依赖图 (Partial Dependence Plots)
    对每个衍生特征，固定其他特征取均值，单独变动目标特征，
    绘制其对预测值的边际效应曲线，揭示非线性定价关系。

Section D · 消融实验 (Ablation Study)
    5 折交叉验证对比：全特征 vs 逐个剔除衍生特征，
    量化每个衍生特征对模型 R² 的增量贡献。

Section E · 综合验证仪表盘
    将 A/B/D 的结果整合为单张 2×3 多面板汇总图。

Author: 毕业设计 · 二手车价格预测系统
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. 学术图表风格配置
# ============================================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_PATH, FIGURES_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# 1. 数据加载与特征矩阵构建
# ============================================================
print("=" * 60)
print("[阶段 0] 加载数据并运行完整预处理 + 特征工程流水线")
print("=" * 60)

raw_df = pd.read_csv(TRAIN_PATH)
y = raw_df['price']
y_log = np.log1p(y)

preprocessor = AdvancedUsedCarPreprocessor()
fe_engineer = HighScoreFeatureEngineer()

X_clean = preprocessor.fit_transform(raw_df.drop(columns=['price']))
X_features = fe_engineer.fit_transform(X_clean, y)

# 分离数值特征矩阵与目标
feature_names = X_features.columns.tolist()
X_matrix = X_features.values.astype(float)
n_samples, n_features = X_matrix.shape

print(f"训练样本数: {n_samples:,}")
print(f"特征维度:   {n_features}")
print(f"特征列表:   {feature_names}")

# ============================================================
# 辅助: 安全日志输出
# ============================================================
def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'))

# ============================================================
# Section A: 排列重要性 (Permutation Importance)
# ============================================================
print("\n" + "=" * 60)
print("[Section A] 排列重要性分析 (Permutation Importance)")
print("=" * 60)

def permutation_importance(model, X, y, feature_names, n_repeats=10, cv=5):
    """
    基于交叉验证的排列重要性计算。

    对每个特征:
      1. 在验证折上打乱该特征列
      2. 计算打乱前后的 R² 差值
      3. 重复 n_repeats 次取均值与标准差

    Returns
    -------
    pd.DataFrame: feature, importance_mean, importance_std
    """
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
                perm_score = model_clone.score(X_val_permuted, y_val)
                scores.append(base_score - perm_score)
            importances[name].extend(scores)

    results = []
    for name in feature_names:
        vals = np.array(importances[name])
        results.append({
            'feature': name,
            'importance_mean': np.mean(vals),
            'importance_std': np.std(vals)
        })
    return pd.DataFrame(results).sort_values('importance_mean', ascending=False)


# 使用 RidgeCV 自动选择最优 alpha
alphas = np.logspace(-2, 3, 20)
ridge_cv = RidgeCV(alphas=alphas)

pi_df = permutation_importance(ridge_cv, X_matrix, y_log.values, feature_names,
                                n_repeats=10, cv=5)

# 打印报告
safe_print(f"\n{'特征':<25} {'重要性均值':>12} {'标准差':>10}")
safe_print("-" * 50)
for _, row in pi_df.iterrows():
    safe_print(f"{row['feature']:<25} {row['importance_mean']:>12.6f} {row['importance_std']:>10.6f}")

# 绘制排列重要性图
fig_a, ax = plt.subplots(figsize=(10, max(5, n_features * 0.45)))
colors_a = ['#2F5496' if v > 0 else '#C00000' for v in pi_df['importance_mean']]

ax.barh(range(len(pi_df)), pi_df['importance_mean'].values, xerr=pi_df['importance_std'].values,
        color=colors_a, edgecolor='white', linewidth=1.2, capsize=4, height=0.6)
ax.set_yticks(range(len(pi_df)))
ax.set_yticklabels(pi_df['feature'].values, fontsize=10)
ax.invert_yaxis()
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Permutation Importance (R^2 decrease)', fontsize=11)
ax.set_title('Section A: Permutation Importance (RidgeCV + 10 repeats x 5-fold CV)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_permutation_importance.png'), dpi=300)
plt.close()
safe_print("\n  [OK] 排列重要性图已保存: reports/figures/validation_permutation_importance.png")

# ============================================================
# Section B: Bootstrap 相关性置信区间
# ============================================================
print("\n" + "=" * 60)
print("[Section B] Bootstrap 相关性置信区间 (B=1000)")
print("=" * 60)

# 提取衍生特征
derived_features = ['annual_milage', 'power_density', 'brand_encoded']
# 也加入基础特征做对比
base_features = ['milage', 'engine_hp', 'engine_liter', 'car_age']
all_validate_features = derived_features + [f for f in base_features if f in feature_names]

B = 1000
bootstrap_results = {}
rng = np.random.RandomState(42)

for feat in all_validate_features:
    if feat not in X_features.columns:
        continue
    x_vals = X_features[feat].values
    pearson_vals = []
    spearman_vals = []
    for _ in range(B):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        # 去重处理: bootstrap 可能采样重复值，corr 需至少 2 个不同值
        try:
            pr, _ = stats.pearsonr(x_vals[idx], y_log.values[idx])
            sr, _ = stats.spearmanr(x_vals[idx], y_log.values[idx])
            pearson_vals.append(pr)
            spearman_vals.append(sr)
        except Exception:
            continue
    pearson_vals = np.array(pearson_vals)
    spearman_vals = np.array(spearman_vals)
    bootstrap_results[feat] = {
        'pearson_mean': np.mean(pearson_vals),
        'pearson_ci_low': np.percentile(pearson_vals, 2.5),
        'pearson_ci_high': np.percentile(pearson_vals, 97.5),
        'spearman_mean': np.mean(spearman_vals),
        'spearman_ci_low': np.percentile(spearman_vals, 2.5),
        'spearman_ci_high': np.percentile(spearman_vals, 97.5),
    }

# 打印报告
safe_print(f"\n{'特征':<20} {'Pearson r':>10} {'95% CI':>22} {'Spearman rho':>13} {'95% CI':>22}")
safe_print("-" * 90)
for feat, res in bootstrap_results.items():
    safe_print(f"{feat:<20} {res['pearson_mean']:>10.4f} "
               f"[{res['pearson_ci_low']:>7.4f}, {res['pearson_ci_high']:>7.4f}]"
               f"  {res['spearman_mean']:>10.4f} "
               f"[{res['spearman_ci_low']:>7.4f}, {res['spearman_ci_high']:>7.4f}]")

# 绘制 Bootstrap CI 对比图
fig_b, axes_b = plt.subplots(1, 2, figsize=(16, max(5, len(all_validate_features) * 0.45)))
feat_list = list(bootstrap_results.keys())

for ax_idx, (metric, title) in enumerate([
    ('pearson', "Pearson r (95% CI)"),
    ('spearman', "Spearman rho (95% CI)")
]):
    ax = axes_b[ax_idx]
    means = [bootstrap_results[f][f'{metric}_mean'] for f in feat_list]
    lows = [bootstrap_results[f][f'{metric}_ci_low'] for f in feat_list]
    highs = [bootstrap_results[f][f'{metric}_ci_high'] for f in feat_list]
    errors_low = [m - l for m, l in zip(means, lows)]
    errors_high = [h - m for m, h in zip(means, highs)]

    y_pos = range(len(feat_list))
    colors_b = ['#ED7D31' if f in derived_features else '#4472C4' for f in feat_list]

    for x, y, xlo, xhi, mc in zip(means, y_pos, errors_low, errors_high, colors_b):
        ax.errorbar(x, y, xerr=[[xlo], [xhi]],
                    fmt='o', capsize=5, capthick=1.5, elinewidth=1.5,
                    color='#333333', markersize=8, markerfacecolor=mc,
                    markeredgecolor='white', markeredgewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_list, fontsize=10)
    ax.set_xlabel(title, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_yaxis()

fig_b.suptitle('Section B: Bootstrap Correlation Confidence Intervals (B=1,000)',
               fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_bootstrap_correlation.png'), dpi=300)
plt.close()
safe_print("\n  [OK] Bootstrap 相关性图已保存: reports/figures/validation_bootstrap_correlation.png")

# ============================================================
# Section C: 偏依赖图 (Partial Dependence Plots)
# ============================================================
print("\n" + "=" * 60)
print("[Section C] 偏依赖图 (Partial Dependence Plots)")
print("=" * 60)

# 基于 RidgeCV 模型计算 PDP
ridge_pdp = RidgeCV(alphas=alphas)
ridge_pdp.fit(X_matrix, y_log.values)

# 只对衍生特征 + 关键基础特征做 PDP
pdp_features = [f for f in all_validate_features if f in feature_names]
pdp_indices = [feature_names.index(f) for f in pdp_features]

n_pdp = len(pdp_features)
n_cols = min(3, n_pdp)
n_rows = int(np.ceil(n_pdp / n_cols))

fig_c, axes_c = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
if n_pdp == 1:
    axes_c = np.array([axes_c])
axes_c = np.atleast_1d(axes_c).flatten()

for i, (feat, feat_idx) in enumerate(zip(pdp_features, pdp_indices)):
    ax = axes_c[i]
    PartialDependenceDisplay.from_estimator(
        ridge_pdp, X_matrix, features=[feat_idx],
        feature_names=feature_names, kind='average', ax=ax,
        grid_resolution=50, line_kw={'color': '#2F5496', 'linewidth': 2},
    )
    # 标注特征类型
    label = '(衍生)' if feat in derived_features else '(基础)'
    ax.set_title(f'{feat} {label}', fontsize=11, fontweight='bold')
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Partial Dependence', fontsize=9)

# 隐藏多余子图
for i in range(n_pdp, len(axes_c)):
    axes_c[i].set_visible(False)

fig_c.suptitle('Section C: Partial Dependence Plots (RidgeCV)',
               fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_partial_dependence.png'), dpi=300)
plt.close()
safe_print("  [OK] 偏依赖图已保存: reports/figures/validation_partial_dependence.png")

# ============================================================
# Section D: 消融实验 (Ablation Study)
# ============================================================
print("\n" + "=" * 60)
print("[Section D] 消融实验 (Ablation Study) — 5-fold CV RidgeCV")
print("=" * 60)

# 基准模型: 全特征
base_scores = cross_val_score(ridge_cv, X_matrix, y_log.values, cv=5,
                               scoring='r2', n_jobs=-1)
safe_print(f"\n全特征基准 R^2 = {base_scores.mean():.4f} +/- {base_scores.std():.4f}")

ablation_results = {}
for drop_feat in derived_features:
    if drop_feat not in feature_names:
        continue
    keep_idx = [i for i, name in enumerate(feature_names) if name != drop_feat]
    X_ablated = X_matrix[:, keep_idx]
    ab_scores = cross_val_score(ridge_cv, X_ablated, y_log.values, cv=5,
                                 scoring='r2', n_jobs=-1)
    delta = base_scores.mean() - ab_scores.mean()
    ablation_results[drop_feat] = {
        'r2_full': base_scores.mean(),
        'r2_ablated': ab_scores.mean(),
        'delta_r2': delta,
        'delta_pct': delta / base_scores.mean() * 100 if base_scores.mean() != 0 else 0
    }
    safe_print(f"剔除 {drop_feat:<20}: R^2 = {ab_scores.mean():.4f} "
               f"| 增量贡献 delta_R^2 = {delta:+.6f} ({ablation_results[drop_feat]['delta_pct']:+.2f}%)")

# 绘制消融实验图
fig_d, ax_d = plt.subplots(figsize=(9, max(4, len(ablation_results) * 0.6)))
feat_d = list(ablation_results.keys())
deltas = [ablation_results[f]['delta_r2'] for f in feat_d]
colors_d = ['#2F5496' if d > 0 else '#C00000' for d in deltas]

bars = ax_d.barh(feat_d, deltas, color=colors_d, edgecolor='white', linewidth=1.2, height=0.5)
ax_d.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
for bar, delta in zip(bars, deltas):
    x_pos = bar.get_width() + 0.0002 if delta >= 0 else bar.get_width() - 0.0008
    ax_d.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{delta:+.6f}',
              va='center', fontsize=11, fontweight='bold')
ax_d.set_xlabel('delta R^2 (full - ablated)', fontsize=11)
ax_d.set_title('Section D: Ablation Study — Feature Incremental Contribution\n'
               '5-fold CV RidgeCV | Positive = feature helps prediction',
               fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_ablation_study.png'), dpi=300)
plt.close()
safe_print("\n  [OK] 消融实验图已保存: reports/figures/validation_ablation_study.png")

# ============================================================
# Section E: 综合验证仪表盘
# ============================================================
print("\n" + "=" * 60)
print("[Section E] 综合验证仪表盘 (2x3 多面板汇总)")
print("=" * 60)

fig_e = plt.figure(figsize=(20, 12))

# ---- 子图 1: 排列重要性 (左上) ----
ax1 = fig_e.add_subplot(2, 3, 1)
colors_a2 = ['#2F5496' if v > 0 else '#C00000' for v in pi_df['importance_mean']]
ax1.barh(range(len(pi_df)), pi_df['importance_mean'].values,
         xerr=pi_df['importance_std'].values, color=colors_a2,
         edgecolor='white', linewidth=1, capsize=3, height=0.6)
ax1.set_yticks(range(len(pi_df)))
ax1.set_yticklabels(pi_df['feature'].values, fontsize=9)
ax1.invert_yaxis()
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_xlabel('R^2 decrease', fontsize=9)
ax1.set_title('A: Permutation Importance', fontsize=11, fontweight='bold')

# ---- 子图 2: Pearson Bootstrap CI (中上) ----
ax2 = fig_e.add_subplot(2, 3, 2)
means_p = [bootstrap_results[f]['pearson_mean'] for f in feat_list]
lows_p = [bootstrap_results[f]['pearson_ci_low'] for f in feat_list]
highs_p = [bootstrap_results[f]['pearson_ci_high'] for f in feat_list]
y_pos2 = range(len(feat_list))
colors_p = ['#ED7D31' if f in derived_features else '#4472C4' for f in feat_list]
for x, y, xlo, xhi, mc in zip(means_p, y_pos2,
                               [m - l for m, l in zip(means_p, lows_p)],
                               [h - m for m, h in zip(means_p, highs_p)],
                               colors_p):
    ax2.errorbar(x, y, xerr=[[xlo], [xhi]],
                 fmt='o', capsize=4, elinewidth=1.2, color='#333333',
                 markersize=7, markerfacecolor=mc, markeredgecolor='white', markeredgewidth=0.8)
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(feat_list, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Pearson r', fontsize=9)
ax2.set_title('B: Bootstrap Pearson r (95% CI)', fontsize=11, fontweight='bold')

# ---- 子图 3: 消融实验 (右上) ----
ax3 = fig_e.add_subplot(2, 3, 3)
feat_d3 = list(ablation_results.keys())
deltas3 = [ablation_results[f]['delta_r2'] for f in feat_d3]
colors_d3 = ['#2F5496' if d > 0 else '#C00000' for d in deltas3]
bars3 = ax3.barh(feat_d3, deltas3, color=colors_d3, edgecolor='white', linewidth=1, height=0.5)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
for bar, delta in zip(bars3, deltas3):
    x_pos = bar.get_width() + 0.0002 if delta >= 0 else bar.get_width() - 0.001
    ax3.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{delta:+.5f}',
             va='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('delta R^2', fontsize=9)
ax3.set_title('D: Ablation Study', fontsize=11, fontweight='bold')

# ---- 子图 4-6: 衍生特征偏依赖图（底行） ----
pdp_derived_only = [f for f in derived_features if f in feature_names]
for i, feat in enumerate(pdp_derived_only):
    ax_pdp = fig_e.add_subplot(2, 3, 4 + i)
    feat_idx = feature_names.index(feat)
    PartialDependenceDisplay.from_estimator(
        ridge_pdp, X_matrix, features=[feat_idx],
        feature_names=feature_names, kind='average', ax=ax_pdp,
        grid_resolution=40, line_kw={'color': '#ED7D31' if feat in derived_features else '#4472C4',
                                      'linewidth': 2},
    )
    ax_pdp.set_title(f'PDP: {feat}', fontsize=10, fontweight='bold')
    ax_pdp.set_xlabel(feat, fontsize=8)
    ax_pdp.set_ylabel('Partial Dependence', fontsize=8)

fig_e.suptitle('Section E: Derived Feature Validation Dashboard\n'
               'Multi-dimensional evidence for feature effectiveness',
               fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_dashboard.png'), dpi=300)
plt.close()
safe_print("  [OK] 综合验证仪表盘已保存: reports/figures/validation_dashboard.png")

# ============================================================
# 验证总结
# ============================================================
print("\n" + "=" * 60)
print("[结论] 衍生特征有效性综合验证汇总")
print("=" * 60)

# 汇总各维度结论
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║              衍生特征多维度验证报告                          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  A. 排列重要性 (Permutation Importance):                     ║
  ║     特征被打乱后模型 R² 下降越多 → 越重要                    ║""")

for _, row in pi_df.iterrows():
    marker = "[衍生]" if row['feature'] in derived_features else "[基础]"
    bar = "█" * max(1, int(row['importance_mean'] / max(pi_df['importance_mean']) * 30))
    print(f"  ║     {marker} {row['feature']:<22} {bar} {row['importance_mean']:.6f}")

print(f"""  ║                                                              ║
  ║  B. Bootstrap 相关性 (B=1,000, 95% CI):                      ║""")
for feat in derived_features:
    if feat in bootstrap_results:
        res = bootstrap_results[feat]
        sig = "显著" if res['pearson_ci_low'] * res['pearson_ci_high'] > 0 else "不显著"
        print(f"  ║     {feat:<25} Pearson r = {res['pearson_mean']:+.4f} "
              f"[{res['pearson_ci_low']:+.4f}, {res['pearson_ci_high']:+.4f}] ({sig})")

print(f"""  ║                                                              ║
  ║  D. 消融实验 (增量贡献 delta_R^2):                            ║""")
for feat in derived_features:
    if feat in ablation_results:
        res = ablation_results[feat]
        print(f"  ║     {feat:<25} delta_R^2 = {res['delta_r2']:+.6f} "
              f"({res['delta_pct']:+.2f}% of full R^2)")

print(f"""  ║                                                              ║
  ║  验证方法升级说明:                                            ║
  ║  · 排列重要性 → 模型级重要性，优于单变量相关                  ║
  ║  · Bootstrap CI → 区间估计，反映统计显著性                    ║
  ║  · 偏依赖图 → 可视化非线性定价轨迹                            ║
  ║  · 消融实验 → 量化增量预测贡献                                ║
  ╚══════════════════════════════════════════════════════════════╝
""")

print("衍生特征有效性综合验证全部完成！")
print(f"输出图表目录: {FIGURES_DIR}")
