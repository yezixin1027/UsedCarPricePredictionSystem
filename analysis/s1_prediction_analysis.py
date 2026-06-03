# analysis/s1_prediction_analysis.py
# ====================================
# 论文 5.1 预测结果分析
#
# 输出:
#   1. 预测值 vs 实际值散点图 (带 R^2 和误差带)
#   2. 残差分布直方图
#   3. 按价格区间的预测表现
#   4. 品牌保值率排名 (基于模型残差)
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH, MODEL_DIR, FIGURES_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_prediction_analysis():
    """运行 5.1 预测结果分析"""
    safe_print("=" * 60)
    safe_print("  5.1 预测结果分析 — Prediction Analysis")
    safe_print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. 加载数据与模型
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price']).values.astype(float)
    y_real = np.expm1(train_df['log_price'].values)

    model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
    if not os.path.exists(model_path):
        safe_print(f"[ERROR] 模型不存在: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 2. 5-fold OOF 预测
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_real))
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = np.log1p(y_real[train_idx])
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = np.expm1(model.predict(X_val))
    oof_preds = np.maximum(oof_preds, 0)

    # 计算指标
    residuals = y_real - oof_preds
    abs_errors = np.abs(residuals)
    mae = mean_absolute_error(y_real, oof_preds)
    rmse = np.sqrt(mean_squared_error(y_real, oof_preds))
    r2 = r2_score(np.log1p(y_real), np.log1p(oof_preds))
    mape = np.mean(abs_errors / np.maximum(y_real, 1)) * 100

    safe_print(f"\n[全局指标 (5-fold OOF)]")
    safe_print(f"  R^2   = {r2:.4f}")
    safe_print(f"  MAE   = {mae:,.0f} 元")
    safe_print(f"  RMSE  = {rmse:,.0f} 元")
    safe_print(f"  MAPE  = {mape:.1f}%")

    # ---- 图 1: 预测值 vs 实际值散点图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 散点图
    max_val = max(y_real.max(), oof_preds.max())
    ax1.scatter(y_real, oof_preds, alpha=0.15, s=2, color='#2F5496', edgecolors='none')
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='完美预测线 (y=x)')
    ax1.fill_between([0, max_val], [0, max_val], [0, max_val * 1.2],
                     alpha=0.1, color='red', label='高估区域')
    ax1.fill_between([0, max_val], [0, max_val], [0, max_val * 0.8],
                     alpha=0.1, color='blue', label='低估区域')
    ax1.set_xlabel('实际价格 (USD)', fontsize=12)
    ax1.set_ylabel('预测价格 (USD)', fontsize=12)
    ax1.set_title(f'图 5-1: 预测值 vs 实际值 (LightGBM 5-fold OOF)\n'
                  f'R^2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, max_val * 1.05)
    ax1.set_ylim(0, max_val * 1.05)
    ax1.grid(True, linestyle=':', alpha=0.3)

    # 残差分布
    ax2.hist(residuals, bins=80, density=True, color='#ED7D31', alpha=0.7, edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.axvline(x=np.mean(residuals), color='#2F5496', linestyle='-', linewidth=1.5,
                label=f'均值残差 = {np.mean(residuals):,.0f}')
    ax2.set_xlabel('残差 (实际 - 预测) USD', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title(f'残差分布 (偏度={pd.Series(residuals).skew():.2f})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    safe_print("  [OK] 预测vs实际散点图已保存")

    # ---- 图 2: 价格区间分析 ----
    bins = [0, 15000, 35000, 75000, np.inf]
    labels = ['低端代步\n(<1.5万)', '中端实用\n(1.5-3.5万)', '高档轻奢\n(3.5-7.5万)', '顶级豪华\n(>7.5万)']
    price_groups = pd.cut(y_real, bins=bins, labels=labels)

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE by price group
    group_mae = [mean_absolute_error(y_real[price_groups == l], oof_preds[price_groups == l])
                 for l in labels]
    axes[0].bar(labels, group_mae, color=['#70AD47', '#2F5496', '#ED7D31', '#C00000'],
                edgecolor='white')
    axes[0].set_title('各价格区间 MAE', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE (元)')
    for i, v in enumerate(group_mae):
        axes[0].text(i, v + 500, f'{v:,.0f}', ha='center', fontweight='bold')

    # MAPE by price group
    group_mape = [np.mean(abs_errors[price_groups == l] / np.maximum(y_real[price_groups == l], 1)) * 100
                  for l in labels]
    axes[1].bar(labels, group_mape, color=['#70AD47', '#2F5496', '#ED7D31', '#C00000'],
                edgecolor='white')
    axes[1].set_title('各价格区间 MAPE', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MAPE (%)')
    for i, v in enumerate(group_mape):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    # 样本占比
    group_counts = price_groups.value_counts()
    axes[2].pie(group_counts.values, labels=group_counts.index, autopct='%1.1f%%',
                colors=['#70AD47', '#2F5496', '#ED7D31', '#C00000'])
    axes[2].set_title('样本量分布', fontsize=12, fontweight='bold')

    fig2.suptitle('图 5-2: 不同价格区间的模型表现', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, 'price_segment_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    safe_print("  [OK] 价格区间分析图已保存")

    # ---- 品牌保值率排名 ----
    safe_print(f"\n[品牌残差分析 — Top-10 品牌保值率]:")
    train_raw = pd.read_csv(PROCESSED_TRAIN_PATH)
    # 从原始数据获取品牌 (需要重新加载)
    from config import TRAIN_PATH
    raw_df = pd.read_csv(TRAIN_PATH)
    analysis_df = raw_df.copy()
    analysis_df['residual'] = residuals
    analysis_df['abs_error'] = abs_errors

    # 样本量>=200的品牌
    brand_counts = analysis_df['brand'].value_counts()
    major_brands = brand_counts[brand_counts >= 200].index
    brand_residuals = analysis_df[analysis_df['brand'].isin(major_brands)].groupby('brand').agg(
        count=('brand', 'count'),
        mean_residual=('residual', 'mean'),
        mae=('abs_error', 'mean'),
        std_residual=('residual', 'std')
    ).sort_values('mean_residual', ascending=False)

    safe_print(f"  {'品牌':<18} {'样本量':>6} {'均残差':>10} {'MAE':>10} {'解读':>20}")
    safe_print(f"  {'─' * 70}")
    for brand, row in brand_residuals.head(10).iterrows():
        interp = '被低估(实际价值更高)' if row['mean_residual'] > 0 else '被高估'
        safe_print(f"  {brand:<18} {int(row['count']):>6} {row['mean_residual']:>10,.0f} "
                   f"{row['mae']:>10,.0f} {interp:>20}")

    safe_print(f"\n  [OK] 5.1 预测结果分析完成")
    return analysis_df


if __name__ == "__main__":
    run_prediction_analysis()
