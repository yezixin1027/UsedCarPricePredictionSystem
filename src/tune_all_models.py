"""
六模型超参数调优 + Stacking 集成脚本
====================================

对 6 个模型分别调优：
  - Ridge:       手动网格搜索 alpha
  - ElasticNet:  手动网格搜索 alpha × l1_ratio
  - Random Forest: 手动网格搜索 n_estimators → max_depth
  - LightGBM:    Optuna 贝叶斯优化 (8参数)
  - KNN:         手动网格搜索 k × weights

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
    safe_print("[1/6] Ridge 岭回归 — 手动网格搜索 alpha")
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
    return {'alpha': best_alpha}, best_r2

    # 画曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas_plot = [r[0] for r in results]
    r2_plot = [r[1] for r in results]
    ax.semilogx(alphas_plot, r2_plot, 'o-', color='#2F5496', linewidth=2, markersize=8)
    ax.set_xlabel('alpha (log scale)', fontsize=12)
    ax.set_ylabel('R2 Score', fontsize=12)
    ax.set_title('Ridge — alpha Hyperparameter Tuning', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tune_ridge_alpha.png'), dpi=300)
    plt.close()

    return {'alpha': best_alpha}, best_r2


# ================================================================
# 2. ElasticNet 手动调优
# ================================================================

def tune_elastic_net(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[2/6] ElasticNet — 手动网格搜索 alpha + l1_ratio")
    safe_print("=" * 60)

    best_r2, best_alpha, best_l1 = -999, 0.1, 0.5
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r2, mae, rmse, mape = cv_evaluate('elastic_net',
                {'alpha': alpha, 'l1_ratio': l1_ratio,
                 'max_iter': 5000, 'random_state': 42}, X, y)
            safe_print(f"  alpha={alpha:<6.2f} l1_ratio={l1_ratio:.1f}  "
                       f"R2={r2:.4f}  MAE={mae:,.0f}")
            if r2 > best_r2:
                best_r2, best_alpha, best_l1 = r2, alpha, l1_ratio

    safe_print(f"  >>> Best: alpha={best_alpha:.2f}, l1_ratio={best_l1:.1f}, R2={best_r2:.4f}")
    return {'alpha': best_alpha, 'l1_ratio': best_l1,
            'max_iter': 5000, 'random_state': 42}, best_r2


# ================================================================
# 3. Random Forest 手动调优
# ================================================================

def tune_random_forest(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[3/6] Random Forest — 手动网格搜索")
    safe_print("=" * 60)

    # 先调 n_estimators
    safe_print("  [Round 1] n_estimators...")
    best_r2, best_n = -999, 200
    for n in [50, 100, 200, 300, 400, 500]:
        r2, mae, rmse, mape = cv_evaluate('random_forest',
            {'n_estimators': n, 'max_depth': 15, 'min_samples_split': 5,
             'random_state': RANDOM_SEED, 'n_jobs': -1}, X, y)
        safe_print(f"    n_estimators={n:<4}  R2={r2:.4f}  MAE={mae:,.0f}")
        if r2 > best_r2:
            best_r2, best_n = r2, n

    # 再调 max_depth
    safe_print(f"  [Round 2] max_depth (n_estimators={best_n})...")
    best_r2_d, best_d = -999, 15
    for d in [5, 8, 10, 12, 15, 20, 25, None]:
        r2, mae, rmse, mape = cv_evaluate('random_forest',
            {'n_estimators': best_n, 'max_depth': d,
             'min_samples_split': 5, 'random_state': RANDOM_SEED, 'n_jobs': -1}, X, y)
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
    return best_params, best_r2_d


# ================================================================
# 4. LightGBM Optuna 调优
# ================================================================

def tune_lightgbm_optuna(X, y, n_trials=50):
    safe_print("\n" + "=" * 60)
    safe_print(f"[4/6] LightGBM — Optuna 贝叶斯优化 ({n_trials} trials)")
    safe_print("=" * 60)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

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
    return best, study.best_value


# ================================================================
# 5. KNN 手动调优
# ================================================================

def tune_knn(X, y):
    safe_print("\n" + "=" * 60)
    safe_print("[5/6] KNN — 手动网格搜索 n_neighbors + weights")
    safe_print("=" * 60)

    best_r2, best_k, best_w = -999, 20, 'distance'
    for k in [5, 10, 15, 20, 30, 50]:
        for w in ['uniform', 'distance']:
            r2, mae, rmse, mape = cv_evaluate('knn',
                {'n_neighbors': k, 'weights': w, 'p': 2, 'n_jobs': -1}, X, y)
            safe_print(f"  k={k:<3} weights={w:<10}  R2={r2:.4f}  MAE={mae:,.0f}")
            if r2 > best_r2:
                best_r2, best_k, best_w = r2, k, w

    safe_print(f"  >>> Best: k={best_k}, weights={best_w}, R2={best_r2:.4f}")
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
    safe_print("  七模型超参数调优 + Stacking 集成")
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

    # ---- 5. XGBoost (树·Boosting) ----
    xgb_params, xgb_r2 = tune_xgboost_optuna(X, y, n_trials=n_trials)
    best_params['xgboost'] = xgb_params

    # ---- 6. CatBoost (树·Boosting) ----
    cat_params, cat_r2 = tune_catboost_optuna(X, y, n_trials=n_trials // 2)
    best_params['catboost'] = cat_params

    # ---- 7. KNN (距离) ----
    knn_params, knn_r2 = tune_knn(X, y)
    best_params['knn'] = knn_params

    # ---- 用最优参数做最终 5-fold CV 评估 ----
    safe_print("\n" + "=" * 70)
    safe_print("  最终评估 (最优参数, 5-fold Stratified CV)")
    safe_print("=" * 70)
    final_results = {}
    model_order = ['ridge', 'elastic_net', 'random_forest',
                   'lightgbm', 'xgboost', 'catboost', 'knn']
    for name in model_order:
        r2, mae, rmse, mape = cv_evaluate(name, best_params[name], X, y)
        final_results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # ---- 8. Stacking (多样化集成) ----
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

    # 替换各模型参数
    for model_name in ['ridge', 'random_forest', 'lightgbm', 'xgboost', 'catboost']:
        params = best_params[model_name]
        import re
        # Find the model block in MODEL_HYPERPARAMS
        pattern = rf'("{model_name}":\s*\{{).*?(\}})'
        replacement = f'"\\1{json.dumps(params, indent=8)}"'
        # This is tricky with regex, use a simpler approach

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
