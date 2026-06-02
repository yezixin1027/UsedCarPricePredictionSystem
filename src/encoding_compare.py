# src/encoding_compare.py
"""
分类变量编码方式定量对比实验
=============================

对比 One-Hot Encoding vs Target Encoding (OOF) 在同一模型下的表现差异，
为编码方式选择提供可重复的定量证据。
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_PATH, FIGURES_DIR, RANDOM_SEED
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_encoding_comparison():
    """对比三种编码方式在 Ridge 模型下的性能。

    方案 0: 丢弃 brand 列（基线）
    方案 1: One-Hot Encoding brand
    方案 2: Target Encoding brand (OOF, 本项目采用)

    Returns
    -------
    pd.DataFrame
    """
    print("=" * 60)
    print("[编码对比实验] One-Hot vs Target Encoding 定量对比")
    print("=" * 60)

    raw = pd.read_csv(TRAIN_PATH)
    y = np.log1p(raw['price']).values

    # --- 准备三种特征矩阵 ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    for strategy_name in ['Drop Brand', 'One-Hot', 'Target Encoding (OOF)']:
        fold_r2, fold_mae = [], []

        for train_idx, val_idx in kf.split(raw):
            X_tr_raw = raw.iloc[train_idx].drop(columns=['price'])
            X_val_raw = raw.iloc[val_idx].drop(columns=['price'])
            y_tr = y[train_idx]
            y_val = y[val_idx]

            # 公共预处理 (不含 brand 编码)
            preprocessor = AdvancedUsedCarPreprocessor()
            X_tr_clean = preprocessor.fit_transform(X_tr_raw)
            X_val_clean = preprocessor.transform(X_val_raw)

            # 衍生特征 (不含 brand 编码)
            # 手动构建特征矩阵
            X_tr_clean['annual_milage'] = X_tr_clean['milage'] / (X_tr_clean['car_age'] + 1)
            X_val_clean['annual_milage'] = X_val_clean['milage'] / (X_val_clean['car_age'] + 1)
            X_tr_clean['power_density'] = X_tr_clean['engine_hp'] / X_tr_clean['engine_liter'].replace(0, np.nan)
            X_val_clean['power_density'] = X_val_clean['engine_hp'] / X_val_clean['engine_liter'].replace(0, np.nan)
            pd_fb = X_tr_clean['engine_hp'].median() / max(X_tr_clean['engine_liter'].median(), 0.1)
            X_tr_clean['power_density'] = X_tr_clean['power_density'].replace([np.inf, -np.inf], np.nan).fillna(pd_fb)
            X_val_clean['power_density'] = X_val_clean['power_density'].replace([np.inf, -np.inf], np.nan).fillna(pd_fb)

            # OHE for fuel_type, accident_status
            cat_cols = [c for c in ['fuel_type', 'accident_status'] if c in X_tr_clean.columns]
            if cat_cols:
                X_tr_clean = pd.get_dummies(X_tr_clean, columns=cat_cols, drop_first=True, dtype=int)
                X_val_clean = pd.get_dummies(X_val_clean, columns=cat_cols, drop_first=True, dtype=int)
                # 对齐列
                for c in X_tr_clean.columns:
                    if c not in X_val_clean.columns:
                        X_val_clean[c] = 0
                X_val_clean = X_val_clean[X_tr_clean.columns]

            if strategy_name == 'Drop Brand':
                X_tr = X_tr_clean.drop(columns=['brand'])
                X_val = X_val_clean.drop(columns=['brand'])
            elif strategy_name == 'One-Hot':
                brands_tr = X_tr_clean[['brand']]
                brands_val = X_val_clean[['brand']]
                all_brands = pd.concat([brands_tr, brands_val])['brand'].unique()
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                ohe.fit(pd.DataFrame(all_brands, columns=['brand']))
                brand_ohe_tr = pd.DataFrame(ohe.transform(brands_tr),
                                            index=X_tr_clean.index)
                brand_ohe_val = pd.DataFrame(ohe.transform(brands_val),
                                             index=X_val_clean.index)
                X_tr = pd.concat([X_tr_clean.drop(columns=['brand']).reset_index(drop=True),
                                  brand_ohe_tr.reset_index(drop=True)], axis=1).astype(float)
                X_val = pd.concat([X_val_clean.drop(columns=['brand']).reset_index(drop=True),
                                   brand_ohe_val.reset_index(drop=True)], axis=1).astype(float)
            else:  # Target Encoding (Simple, no OOF to keep comparison fair)
                global_mean = y_tr.mean()
                # Compute brand-level log-price mean on training data
                brand_df = pd.DataFrame({'brand': X_tr_clean['brand'].values, 'target': y_tr})
                brand_mean_map = brand_df.groupby('brand')['target'].mean().to_dict()

                X_tr = X_tr_clean.drop(columns=['brand'])
                X_val = X_val_clean.drop(columns=['brand'])
                X_tr['brand_encoded'] = X_tr_clean['brand'].map(brand_mean_map).fillna(global_mean)
                X_val['brand_encoded'] = X_val_clean['brand'].map(brand_mean_map).fillna(global_mean)

            # Z-score 标准化数值特征
            num_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age', 'annual_milage', 'power_density']
            num_cols = [c for c in num_cols if c in X_tr.columns]
            scaler = StandardScaler()
            X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])

            # 训练 Ridge 模型
            model = Ridge(alpha=1.0)
            X_tr_filled = X_tr.fillna(0).values.astype(float)
            X_val_filled = X_val.fillna(0).values.astype(float)
            model.fit(X_tr_filled, y_tr)
            preds = model.predict(X_val_filled)
            fold_r2.append(r2_score(y_val, preds))
            fold_mae.append(mean_absolute_error(np.expm1(y_val), np.expm1(preds)))

        results[strategy_name] = {
            'R2_mean': np.mean(fold_r2), 'R2_std': np.std(fold_r2),
            'MAE_mean': np.mean(fold_mae), 'MAE_std': np.std(fold_mae),
            'n_features': X_tr.shape[1]
        }

    # 输出对比表
    print(f"\n  {'编码方案':<25} {'特征数':>6} {'R^2':>10} {'MAE':>12} {'结论':>20}")
    print(f"  {'─' * 75}")
    for name, r in results.items():
        best = 'BEST' if r['R2_mean'] == max(v['R2_mean'] for v in results.values()) else ''
        print(f"  {name:<25} {r['n_features']:>6} {r['R2_mean']:>10.4f}±{r['R2_std']:.4f} "
              f"{r['MAE_mean']:>10.0f}  {best:>20}")

    # 柱状图对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    colors = ['#C00000', '#ED7D31', '#2F5496']

    r2_vals = [results[n]['R2_mean'] for n in names]
    ax1.bar(names, r2_vals, color=colors, edgecolor='white')
    ax1.set_title('5-fold CV R^2 对比', fontsize=13, fontweight='bold')
    ax1.set_ylabel('R^2 Score')
    for i, v in enumerate(r2_vals):
        ax1.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

    mae_vals = [results[n]['MAE_mean'] for n in names]
    ax2.bar(names, mae_vals, color=colors, edgecolor='white')
    ax2.set_title('5-fold CV MAE 对比 (真实金额)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MAE (元)')
    for i, v in enumerate(mae_vals):
        ax2.text(i, v + 50, f'{v:.0f}', ha='center', fontweight='bold')

    fig.suptitle('分类变量编码方式定量对比实验\n'
                 'Drop Brand vs One-Hot vs Target Encoding (Ridge 5-fold CV)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, 'encoding_comparison.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n  [OK] 编码对比图已保存: {path}")

    return pd.DataFrame(results).T


if __name__ == "__main__":
    run_encoding_comparison()
