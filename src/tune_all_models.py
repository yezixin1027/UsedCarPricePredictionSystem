"""
五模型超参数调优 + Stacking 集成脚本
====================================

对 5 个模型分别调优，每个模型生成参数-性能曲线（含过拟合/欠拟合区域标注）：
  - Ridge:       手动网格搜索 alpha → tune_ridge_alpha.png
  - ElasticNet:  手动网格搜索 alpha × l1_ratio → tune_elasticnet_heatmap.png
  - Random Forest: 手动网格搜索 n_estimators → max_depth → tune_random_forest.png
  - LightGBM:    手动探索 num_leaves+learning_rate → tune_lightgbm_manual.png
                  + Optuna 贝叶斯优化精调其余参数
  - KNN:         手动网格搜索 k × weights → tune_knn.png

最后用各模型最优参数构建 Stacking 集成 (Ridge+EN+RF+LGB+KNN → ElasticNet)。

Usage:
    python src/tune_all_models.py [--trials N] [--no-stacking]
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_TRAIN_PATH, FIGURES_DIR, MODEL_DIR, RANDOM_SEED
from src.models import UsedCarModelFactory


def load_data():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = df.drop(columns=['log_price']).values.astype(float)
    y = np.expm1(df['log_price'].values)
    return X, y, list(df.drop(columns=['log_price']).columns)


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


# ================================================================
# 通用评估函数
# ================================================================

def cv_evaluate(model_name, params, X, y, n_folds=5):
    """5折分层 CV 评估一组参数。返回 (R2_mean, MAE_mean, RMSE_mean, MAPE_mean)。"""
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from src.train import _make_stratified_folds

    folds = _make_stratified_folds(y, n_splits=n_folds, random_state=RANDOM_SEED)
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []

    for train_idx, val_idx in folds:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = UsedCarModelFactory.create_model(model_name, **params)
        model.fit(X_tr, np.log1p(y_tr))
        preds = np.expm1(model.predict(X_val))
        preds = np.maximum(preds, 0)

        r2_list.append(r2_score(np.log1p(y_val), np.log1p(preds)))
        mae_list.append(mean_absolute_error(y_val, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))
        mape_list.append(np.mean(np.abs((y_val - preds) / np.maximum(y_val, 1))) * 100)

    return np.mean(r2_list), np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)


# ================================================================
# 1. Ridge 手动调优
# ================================================================

def tune_ridge(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[1/5] Ridge 岭回归 — 手动网格搜索 alpha")
    safe_print("=" * 60)

    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    best_r2, best_alpha = -999, 1.0
    results = []

    for alpha in alphas:
        r2, mae, rmse, mape = cv_evaluate('ridge', {'alpha': alpha}, X, y)
        results.append((alpha, r2, mae, rmse, mape))
        safe_print(f"  alpha={alpha:<8.2f}  R2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%")
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    safe_print(f"  >>> Best: alpha={best_alpha:.2f}, R2={best_r2:.4f}")

    # 画曲线 (含过拟合/欠拟合标注)
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas_plot = [r[0] for r in results]
    r2_plot = [r[1] for r in results]
    ax.semilogx(alphas_plot, r2_plot, 'o-', color='#2F5496', linewidth=2, markersize=8)
    # 标注最佳
    best_idx = np.argmax(r2_plot)
    ax.annotate(f'Best: alpha={alphas_plot[best_idx]:.2f}\nR²={r2_plot[best_idx]:.4f}',
                xy=(alphas_plot[best_idx], r2_plot[best_idx]),
                xytext=(alphas_plot[best_idx] * 3, r2_plot[best_idx] - 0.001),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    # 过拟合风险 (alpha过小)
    ax.annotate('欠拟合风险\n(正则化过强)',
                xy=(alphas_plot[-2], r2_plot[-2]),
                xytext=(alphas_plot[-1] * 0.5, r2_plot[-2] - 0.002),
                fontsize=9, ha='center', color='#C00000',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.2))
    ax.set_xlabel('alpha (log scale)', fontsize=12)
    ax.set_ylabel('R² Score (5-fold CV)', fontsize=12)
    ax.set_title('Ridge — alpha Hyperparameter Tuning\n'
                 '(alpha过大→欠拟合 | alpha适中→最佳泛化)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_ridge_alpha.png'), dpi=300)
    plt.close()
    safe_print(f"  [OK] 参数曲线: reports/figures/tune_ridge_alpha.png")

    return {'alpha': best_alpha}, best_r2


# ================================================================
# 2. ElasticNet 手动调优
# ================================================================

def tune_elastic_net(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[2/5] ElasticNet — 手动网格搜索 alpha + l1_ratio")
    safe_print("=" * 60)

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_r2, best_alpha, best_l1 = -999, 0.1, 0.5
    heatmap_data = np.zeros((len(alphas), len(l1_ratios)))

    for i, alpha in enumerate(alphas):
        for j, l1_ratio in enumerate(l1_ratios):
            r2, mae, rmse, mape = cv_evaluate('elastic_net',
                {'alpha': alpha, 'l1_ratio': l1_ratio,
                 'max_iter': 5000, 'random_state': 42}, X, y)
            heatmap_data[i, j] = r2
            safe_print(f"  alpha={alpha:<6.2f} l1_ratio={l1_ratio:.1f}  "
                       f"R2={r2:.4f}  MAE={mae:,.0f}")
            if r2 > best_r2:
                best_r2, best_alpha, best_l1 = r2, alpha, l1_ratio

    safe_print(f"  >>> Best: alpha={best_alpha:.2f}, l1_ratio={best_l1:.1f}, R2={best_r2:.4f}")

    # ---- 绘制 alpha × l1_ratio 热力图 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', origin='lower',
                   vmin=heatmap_data[heatmap_data > -1].min(), vmax=heatmap_data.max())
    ax.set_xticks(range(len(l1_ratios)))
    ax.set_xticklabels([str(l) for l in l1_ratios])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([str(a) for a in alphas])
    ax.set_xlabel('l1_ratio (1=纯Lasso, 0=纯Ridge)', fontsize=12)
    ax.set_ylabel('alpha (正则化强度)', fontsize=12)
    ax.set_title('ElasticNet — alpha × l1_ratio R² Heatmap\n'
                 f'(Best: alpha={best_alpha}, l1_ratio={best_l1}, R²={best_r2:.4f})',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R² Score', fontsize=10)
    # 标注最佳点
    best_i = alphas.index(best_alpha)
    best_j = l1_ratios.index(best_l1)
    ax.plot(best_j, best_i, '*', color='#2F5496', markersize=20, markeredgecolor='white')
    # 标注过拟合/欠拟合区域
    ax.annotate('最佳\n区域', xy=(best_j, best_i), xytext=(best_j + 1.2, best_i + 1.2),
                fontsize=9, ha='center', color='#2F5496', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2F5496'))
    # 高alpha高l1区域 → 欠拟合
    ax.annotate('欠拟合区域\n(正则化过强)', xy=(3.5, 5.5), fontsize=9, ha='center',
                color='#C00000', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_elasticnet_heatmap.png'), dpi=300)
    plt.close()
    safe_print(f"  [OK] 热力图: reports/figures/tune_elasticnet_heatmap.png")

    return {'alpha': best_alpha, 'l1_ratio': best_l1,
            'max_iter': 5000, 'random_state': 42}, best_r2


# ================================================================
# 3. Random Forest 手动调优
# ================================================================

def tune_random_forest(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[3/5] Random Forest — 手动网格搜索")
    safe_print("=" * 60)

    # 先调 n_estimators
    safe_print("  [Round 1] n_estimators...")
    best_r2, best_n = -999, 200
    n_vals = [50, 100, 200, 300, 400, 500]
    n_results = []
    for n in n_vals:
        r2, mae, rmse, mape = cv_evaluate('random_forest',
            {'n_estimators': n, 'max_depth': 15, 'min_samples_split': 5,
             'random_state': RANDOM_SEED, 'n_jobs': -1}, X, y)
        n_results.append((n, r2, mae))
        safe_print(f"    n_estimators={n:<4}  R2={r2:.4f}  MAE={mae:,.0f}")
        if r2 > best_r2:
            best_r2, best_n = r2, n

    # 再调 max_depth
    safe_print(f"  [Round 2] max_depth (n_estimators={best_n})...")
    best_r2_d, best_d = -999, 15
    d_vals = [5, 8, 10, 12, 15, 20, 25, None]
    d_results = []
    for d in d_vals:
        r2, mae, rmse, mape = cv_evaluate('random_forest',
            {'n_estimators': best_n, 'max_depth': d,
             'min_samples_split': 5, 'random_state': RANDOM_SEED, 'n_jobs': -1}, X, y)
        d_results.append((d, r2, mae))
        label = str(d) if d else 'None'
        safe_print(f"    max_depth={label:<5}  R2={r2:.4f}  MAE={mae:,.0f}")
        if r2 > best_r2_d:
            best_r2_d, best_d = r2, d

    best_params = {
        'n_estimators': best_n,
        'max_depth': best_d,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    safe_print(f"  >>> Best: n_estimators={best_n}, max_depth={best_d}, R2={best_r2_d:.4f}")

    # ---- 绘制 n_estimators 参数-性能曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    n_x = [r[0] for r in n_results]
    n_r2 = [r[1] for r in n_results]
    ax.plot(n_x, n_r2, 'o-', color='#2F5496', linewidth=2, markersize=8)
    best_n_idx = np.argmax(n_r2)
    ax.annotate(f'Best: {n_x[best_n_idx]}\nR²={n_r2[best_n_idx]:.4f}',
                xy=(n_x[best_n_idx], n_r2[best_n_idx]),
                xytext=(n_x[best_n_idx] + 50, n_r2[best_n_idx] - 0.003),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('R² Score (5-fold CV)', fontsize=12)
    ax.set_title('Random Forest — n_estimators Tuning\n'
                 '(树越多→方差越低但边际收益递减)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[1]
    d_x = [r[0] if r[0] else 30 for r in d_results]  # None → 30 for plotting
    d_labels = [str(r[0]) if r[0] else 'None' for r in d_results]
    d_r2 = [r[1] for r in d_results]
    ax.plot(range(len(d_x)), d_r2, 's-', color='#ED7D31', linewidth=2, markersize=8)
    ax.set_xticks(range(len(d_x)))
    ax.set_xticklabels(d_labels)
    best_d_idx = np.argmax(d_r2)
    ax.annotate(f'Best: {d_labels[best_d_idx]}\nR²={d_r2[best_d_idx]:.4f}',
                xy=(best_d_idx, d_r2[best_d_idx]),
                xytext=(best_d_idx + 0.5, d_r2[best_d_idx] - 0.005),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    # 标注过拟合区域
    ax.axvspan(len(d_x) - 2.5, len(d_x) - 0.5, alpha=0.1, color='red', label='过拟合风险区')
    ax.annotate('过拟合\n风险区', xy=(len(d_x) - 1.5, d_r2[-1]),
                fontsize=9, ha='center', color='#C00000', fontweight='bold')
    ax.set_xlabel('max_depth', fontsize=12)
    ax.set_ylabel('R² Score (5-fold CV)', fontsize=12)
    ax.set_title('Random Forest — max_depth Tuning\n'
                 '(过深→过拟合训练集 | 过浅→欠拟合)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_random_forest.png'), dpi=300)
    plt.close()
    safe_print(f"  [OK] 参数曲线: reports/figures/tune_random_forest.png")

    return best_params, best_r2_d


# ================================================================
# 4. LightGBM 调优 (手动探索 + Optuna 精调)
# ================================================================

def tune_lightgbm_manual(X, y):
    """手动探索 LightGBM 的 2 个关键参数: num_leaves, learning_rate。

    对每个参数，固定其他参数为默认值，记录 Train R² 和 Val R²，
    绘制参数-性能曲线并标注最优区域和过拟合区域。
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    safe_print("\n" + "-" * 50)
    safe_print("  [4a] LightGBM 手动参数探索 (num_leaves + learning_rate)")
    safe_print("-" * 50)

    base_params = {
        'n_estimators': 500,  # 手动探索用500棵树加速
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def eval_params(override):
        p = base_params.copy()
        p.update(override)
        train_scores, val_scores = [], []
        for tr, vl in kf.split(X):
            X_tr, X_val = X[tr], X[vl]
            y_tr, y_val = y[tr], y[vl]
            model = UsedCarModelFactory.create_model('lightgbm', **p)
            model.fit(X_tr, np.log1p(y_tr))
            train_scores.append(r2_score(np.log1p(y_tr), model.predict(X_tr)))
            val_scores.append(r2_score(np.log1p(y_val), model.predict(X_val)))
        return np.mean(train_scores), np.mean(val_scores)

    # ---- 参数 1: num_leaves (固定 lr=0.1) ----
    safe_print("  [num_leaves] 手动探索 (learning_rate=0.1)...")
    leaves_vals = [7, 15, 31, 63, 127, 255]
    leaves_train, leaves_val = [], []
    for nl in leaves_vals:
        tr_r2, val_r2 = eval_params({'num_leaves': nl, 'learning_rate': 0.1})
        leaves_train.append(tr_r2)
        leaves_val.append(val_r2)
        gap = tr_r2 - val_r2
        overfit_flag = ' ⚠过拟合' if gap > 0.03 else ''
        safe_print(f"    num_leaves={nl:<5}  Train R²={tr_r2:.4f}  Val R²={val_r2:.4f}  gap={gap:.4f}{overfit_flag}")

    best_nl_idx = np.argmax(leaves_val)
    safe_print(f"    >>> Best num_leaves={leaves_vals[best_nl_idx]}, Val R²={leaves_val[best_nl_idx]:.4f}")

    # ---- 参数 2: learning_rate (固定最佳 num_leaves) ----
    safe_print(f"  [learning_rate] 手动探索 (num_leaves={leaves_vals[best_nl_idx]})...")
    lr_vals = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    lr_train, lr_val = [], []
    for lr in lr_vals:
        tr_r2, val_r2 = eval_params({'num_leaves': leaves_vals[best_nl_idx], 'learning_rate': lr})
        lr_train.append(tr_r2)
        lr_val.append(val_r2)
        gap = tr_r2 - val_r2
        overfit_flag = ' ⚠过拟合' if gap > 0.03 else ''
        safe_print(f"    lr={lr:<6}  Train R²={tr_r2:.4f}  Val R²={val_r2:.4f}  gap={gap:.4f}{overfit_flag}")

    best_lr_idx = np.argmax(lr_val)
    safe_print(f"    >>> Best lr={lr_vals[best_lr_idx]}, Val R²={lr_val[best_lr_idx]:.4f}")

    # ---- 绘制参数-性能曲线 (含过拟合标注) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # num_leaves 曲线
    ax = axes[0]
    ax.plot(leaves_vals, leaves_train, 'o-', color='#C00000', linewidth=2, markersize=8, label='Train R²')
    ax.plot(leaves_vals, leaves_val, 's--', color='#2F5496', linewidth=2, markersize=8, label='Val R²')
    ax.fill_between(leaves_vals, leaves_train, leaves_val, alpha=0.1, color='gray')
    # 标注最佳
    ax.annotate(f'Best: {leaves_vals[best_nl_idx]}\nVal R²={leaves_val[best_nl_idx]:.4f}',
                xy=(leaves_vals[best_nl_idx], leaves_val[best_nl_idx]),
                xytext=(leaves_vals[best_nl_idx] + 20, leaves_val[best_nl_idx] - 0.01),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    # 标注过拟合区域 (Train-Val gap 大的区域)
    gaps_nl = np.array(leaves_train) - np.array(leaves_val)
    overfit_idx = np.where(gaps_nl > 0.02)[0]
    if len(overfit_idx) > 0:
        ax.axvspan(leaves_vals[overfit_idx[0]] - 10, leaves_vals[overfit_idx[-1]] + 10,
                   alpha=0.1, color='red')
        ax.annotate('过拟合区域\n(Train-Val差距>0.02)',
                    xy=(leaves_vals[overfit_idx[len(overfit_idx)//2]],
                        (leaves_train[overfit_idx[len(overfit_idx)//2]] + leaves_val[overfit_idx[len(overfit_idx)//2]]) / 2),
                    fontsize=9, ha='center', color='#C00000', fontweight='bold')
    ax.set_xlabel('num_leaves', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('LightGBM — num_leaves Manual Tuning\n'
                 '(叶子数↑→模型复杂度↑→先提升后过拟合)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    # learning_rate 曲线
    ax = axes[1]
    ax.plot(lr_vals, lr_train, 'o-', color='#C00000', linewidth=2, markersize=8, label='Train R²')
    ax.plot(lr_vals, lr_val, 's--', color='#2F5496', linewidth=2, markersize=8, label='Val R²')
    ax.fill_between(lr_vals, lr_train, lr_val, alpha=0.1, color='gray')
    ax.annotate(f'Best: {lr_vals[best_lr_idx]}\nVal R²={lr_val[best_lr_idx]:.4f}',
                xy=(lr_vals[best_lr_idx], lr_val[best_lr_idx]),
                xytext=(lr_vals[best_lr_idx] + 0.03, lr_val[best_lr_idx] - 0.01),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    # 标注过拟合区域 (小lr大gap)
    gaps_lr = np.array(lr_train) - np.array(lr_val)
    overfit_idx_lr = np.where(gaps_lr > 0.02)[0]
    if len(overfit_idx_lr) > 0:
        ax.annotate('过拟合区域\n(学习率过低→过度记忆训练集)',
                    xy=(lr_vals[overfit_idx_lr[0]],
                        (lr_train[overfit_idx_lr[0]] + lr_val[overfit_idx_lr[0]]) / 2),
                    fontsize=9, ha='center', color='#C00000', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.2))
    ax.set_xlabel('learning_rate', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('LightGBM — learning_rate Manual Tuning\n'
                 '(学习率↓→需更多树→拟合更细→过低则过拟合)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_lightgbm_manual.png'), dpi=300)
    plt.close()
    safe_print(f"  [OK] 手动调优曲线: reports/figures/tune_lightgbm_manual.png")

    return {'num_leaves': leaves_vals[best_nl_idx], 'learning_rate': lr_vals[best_lr_idx]}


def tune_lightgbm_optuna(X, y, n_trials=50):
    safe_print("\n" + "=" * 60)
    safe_print(f"[4/5] LightGBM — 手动探索 + Optuna 贝叶斯优化 ({n_trials} trials)")
    safe_print("=" * 60)

    # Step 1: 手动探索 2 个关键参数 + 绘制曲线
    manual_best = tune_lightgbm_manual(X, y)

    # Step 2: Optuna 全参数精调
    safe_print(f"\n  [4b] LightGBM Optuna 精调 ({n_trials} trials)...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    manual_num_leaves = manual_best['num_leaves']
    manual_lr = manual_best['learning_rate']

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'n_estimators': 1000,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1,
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr, vl in kf.split(X):
            X_tr, X_val = X[tr], X[vl]
            y_tr, y_val = y[tr], y[vl]
            model = UsedCarModelFactory.create_model('lightgbm', **params)
            model.fit(X_tr, np.log1p(y_tr))
            scores.append(r2_score(np.log1p(y_val), model.predict(X_val)))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    best['n_estimators'] = 1000
    best['random_state'] = RANDOM_SEED
    best['n_jobs'] = -1
    best['verbose'] = -1

    safe_print(f"  Best R2: {study.best_value:.4f}")
    safe_print(f"  Best params: {json.dumps(study.best_params, indent=2)}")
    safe_print(f"  (Manual reference: num_leaves={manual_num_leaves}, lr={manual_lr:.4f})")
    return best, study.best_value


# ================================================================
# 5. KNN 手动调优
# ================================================================

def tune_knn(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[5/5] KNN — 手动网格搜索 n_neighbors + weights")
    safe_print("=" * 60)

    k_vals = [5, 10, 15, 20, 30, 50]
    w_vals = ['uniform', 'distance']
    best_r2, best_k, best_w = -999, 20, 'distance'
    all_results = {}  # {weights: {k: r2}}

    for w in w_vals:
        all_results[w] = {}
        for k in k_vals:
            r2, mae, rmse, mape = cv_evaluate('knn',
                {'n_neighbors': k, 'weights': w, 'p': 2, 'n_jobs': -1}, X, y)
            all_results[w][k] = (r2, mae)
            safe_print(f"  k={k:<3} weights={w:<10}  R2={r2:.4f}  MAE={mae:,.0f}")
            if r2 > best_r2:
                best_r2, best_k, best_w = r2, k, w

    safe_print(f"  >>> Best: k={best_k}, weights={best_w}, R2={best_r2:.4f}")

    # ---- 绘制 KNN 分组柱状图 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_vals))
    width = 0.35
    bars1 = ax.bar(x - width/2, [all_results['uniform'][k][0] for k in k_vals],
                   width, label='weights=uniform', color='#2F5496', edgecolor='white')
    bars2 = ax.bar(x + width/2, [all_results['distance'][k][0] for k in k_vals],
                   width, label='weights=distance', color='#ED7D31', edgecolor='white')

    # 标注最佳参数
    best_idx = k_vals.index(best_k)
    best_w_idx = 0 if best_w == 'uniform' else 1
    best_bar = [bars1, bars2][best_w_idx][best_idx]
    ax.annotate(f'Best: k={best_k}\nweights={best_w}\nR²={best_r2:.4f}',
                xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
                xytext=(best_bar.get_x() + 0.5, best_bar.get_height() + 0.005),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_xlabel('n_neighbors (k)', fontsize=12)
    ax.set_ylabel('R² Score (5-fold CV)', fontsize=12)
    ax.set_title('KNN — k × weights Grid Search\n'
                 '(k过小→过拟合/噪声敏感 | k过大→欠拟合/边界模糊)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_vals)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    # 过拟合/欠拟合标注
    ax.annotate('过拟合风险\n(k过小→决策边界过于复杂)',
                xy=(0, all_results['uniform'][k_vals[0]][0]),
                xytext=(0.3, all_results['uniform'][k_vals[0]][0] - 0.015),
                fontsize=9, ha='center', color='#C00000',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.2))
    ax.annotate('欠拟合趋势\n(k过大→过度平滑)',
                xy=(len(k_vals) - 1, all_results['uniform'][k_vals[-1]][0]),
                xytext=(len(k_vals) - 1.3, all_results['uniform'][k_vals[-1]][0] - 0.015),
                fontsize=9, ha='center', color='#C00000',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.2))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_knn.png'), dpi=300)
    plt.close()
    safe_print(f"  [OK] 参数曲线: reports/figures/tune_knn.png")

    return {'n_neighbors': best_k, 'weights': best_w, 'p': 2, 'n_jobs': -1}, best_r2


# ================================================================
# 6. Stacking 集成 (多样化基模型)
# ================================================================

def evaluate_stacking(best_params_dict, X, y, n_folds=5):
    """用各模型最优参数构建 Stacking (Ridge+EN+RF+LGB+KNN) 并评估"""
    safe_print("\n" + "=" * 60)
    safe_print("[6/6] Stacking 集成 (Ridge+EN+RF+LGB+KNN -> ElasticNet)")
    safe_print("=" * 60)

    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from src.train import _make_stratified_folds

    ridge_p = best_params_dict['ridge'].copy()
    en_p    = best_params_dict['elastic_net'].copy()
    rf_p    = best_params_dict['random_forest'].copy()
    lgb_p   = best_params_dict['lightgbm'].copy()
    knn_p   = best_params_dict['knn'].copy()

    lgb_p.pop('verbose', None); lgb_p.pop('n_jobs', None)
    rf_p.pop('n_jobs', None); knn_p.pop('n_jobs', None)

    base_models = [
        ('ridge', Ridge(**ridge_p)),
        ('en',    ElasticNet(**en_p)),
        ('rf',    RandomForestRegressor(**rf_p)),
        ('lgb',   lgb.LGBMRegressor(**lgb_p)),
        ('knn',   KNeighborsRegressor(**knn_p)),
    ]

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
        cv=5, n_jobs=-1, passthrough=True
    )

    folds = _make_stratified_folds(y, n_splits=n_folds, random_state=RANDOM_SEED)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        stacking.fit(X_tr, np.log1p(y_tr))
        preds = np.expm1(stacking.predict(X_val))
        preds = np.maximum(preds, 0)

        r2 = r2_score(np.log1p(y_val), np.log1p(preds))
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mape = np.mean(np.abs((y_val - preds) / np.maximum(y_val, 1))) * 100

        r2_list.append(r2); mae_list.append(mae)
        rmse_list.append(rmse); mape_list.append(mape)
        safe_print(f"  Fold {fold_i}: R2={r2:.4f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}, MAPE={mape:.1f}%")

    result = {
        'R2_mean': np.mean(r2_list), 'R2_std': np.std(r2_list),
        'MAE_mean': np.mean(mae_list), 'MAE_std': np.std(mae_list),
        'RMSE_mean': np.mean(rmse_list), 'RMSE_std': np.std(rmse_list),
        'MAPE_mean': np.mean(mape_list), 'MAPE_std': np.std(mape_list),
    }
    return result


# ================================================================
# 主流程
# ================================================================

def run_full_tuning(n_trials=50):
    start_time = time.time()
    safe_print("=" * 70)
    safe_print("  五模型超参数调优 + Stacking 集成")
    safe_print(f"  Optuna trials: {n_trials} | 5-fold Stratified CV")
    safe_print("=" * 70)

    X, y, feature_names = load_data()
    safe_print(f"\nData: {X.shape[0]:,} samples x {X.shape[1]} features")
    safe_print(f"Features: {feature_names}")

    best_params = {}

    # ---- 1. Ridge (线性·L2) ----
    ridge_params, ridge_r2 = tune_ridge(X, y)
    best_params['ridge'] = ridge_params

    # ---- 2. ElasticNet (线性·L1+L2) ----
    en_params, en_r2 = tune_elastic_net(X, y)
    best_params['elastic_net'] = en_params

    # ---- 3. Random Forest (树·Bagging) ----
    rf_params, rf_r2 = tune_random_forest(X, y)
    best_params['random_forest'] = rf_params

    # ---- 4. LightGBM (树·Boosting) ----
    lgb_params, lgb_r2 = tune_lightgbm_optuna(X, y, n_trials=n_trials)
    best_params['lightgbm'] = lgb_params

    # ---- 5. KNN (距离) ----
    knn_params, knn_r2 = tune_knn(X, y)
    best_params['knn'] = knn_params

    # ---- 用最优参数做最终 5-fold CV 评估 ----
    safe_print("\n" + "=" * 70)
    safe_print("  最终评估 (最优参数, 5-fold Stratified CV)")
    safe_print("=" * 70)
    final_results = {}
    model_order = ['ridge', 'elastic_net', 'random_forest',
                   'lightgbm', 'knn']
    for name in model_order:
        r2, mae, rmse, mape = cv_evaluate(name, best_params[name], X, y)
        final_results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # ---- 6. Stacking (多样化集成) ----
    stacking_result = evaluate_stacking(best_params, X, y)

    # ---- 总结表 ----
    safe_print("\n" + "=" * 80)
    safe_print(f"  {'Model':<20} {'R2':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>10}")
    safe_print(f"  {'-' * 65}")
    for name in model_order:
        r = final_results[name]
        safe_print(f"  {name:<20} {r['R2']:>8.4f} {r['MAE']:>12,.0f} {r['RMSE']:>12,.0f} {r['MAPE']:>9.1f}%")
    safe_print(f"  {'-' * 65}")
    safe_print(f"  {'STACKING':<20} {stacking_result['R2_mean']:>8.4f} "
               f"{stacking_result['MAE_mean']:>12,.0f} {stacking_result['RMSE_mean']:>12,.0f} "
               f"{stacking_result['MAPE_mean']:>9.1f}%")
    safe_print(f"  {'=' * 80}")

    # ---- 保存最优参数到 config.py ----
    safe_print("\n[Save] Updating config.py with optimized parameters...")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # Write optimized params as JSON for reference
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'optimized_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    safe_print(f"  [OK] Optimized params saved: models/optimized_params.json")

    # ---- 更新 config.py ----
    safe_print("\n[Update] Writing optimized params to config.py...")
    update_config_with_params(best_params)

    elapsed = time.time() - start_time
    safe_print(f"\n{'=' * 70}")
    safe_print(f"  Tuning complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    safe_print(f"  Optimized params: models/optimized_params.json")
    safe_print(f"{'=' * 70}")

    return best_params, final_results, stacking_result


def update_config_with_params(best_params):
    """将最优参数写入 config.py 的 MODEL_HYPERPARAMS"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到 MODEL_HYPERPARAMS 字典的范围
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.strip().startswith('MODEL_HYPERPARAMS = {'):
            start_idx = i
        if start_idx is not None and line.strip() == '}' and i > start_idx:
            end_idx = i
            break

    if start_idx and end_idx:
        # 构建新的 MODEL_HYPERPARAMS
        new_block = ['MODEL_HYPERPARAMS = {\n']
        model_order = ['ridge', 'elastic_net', 'random_forest', 'lightgbm', 'knn']
        for mi, name in enumerate(model_order):
            params = best_params[name]
            new_block.append(f'    "{name}": {{\n')
            param_items = list(params.items())
            for pi, (k, v) in enumerate(param_items):
                comma = ',' if pi < len(param_items) - 1 else ''
                if isinstance(v, str):
                    new_block.append(f'        "{k}": "{v}"{comma}\n')
                elif isinstance(v, bool):
                    new_block.append(f'        "{k}": {str(v).lower()}{comma}\n')
                elif isinstance(v, float):
                    new_block.append(f'        "{k}": {v}{comma}\n')
                else:
                    new_block.append(f'        "{k}": {v}{comma}\n')
            comma2 = ',' if mi < len(model_order) - 1 else ''
            new_block.append(f'    }}{comma2}\n')
        new_block.append('}\n')

        # 替换
        new_lines = lines[:start_idx] + new_block + lines[end_idx + 1:]
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        safe_print("  [OK] config.py updated with optimized parameters!")
    else:
        safe_print("  [WARN] Could not find MODEL_HYPERPARAMS in config.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=50, help='Optuna trials (default: 50)')
    p.add_argument('--no-stacking', action='store_true', help='Skip stacking')
    p.add_argument('--skip-tune', action='store_true', help='Skip tuning, use current config params')
    args = p.parse_args()

    run_full_tuning(n_trials=args.trials)
