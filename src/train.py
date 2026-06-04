# src/train.py
"""
模型训练与比较模块 (增强版)
============================

提供统一的多模型训练、分层交叉验证、对比分析、Stacking 集成功能。

Classes:
    ModelTrainer -- 统一训练器，封装模型训练/预测/CV/持久化

Functions:
    train_all_models -- 分层 CV 拆分下训练所有模型并返回比较 DataFrame
    train_stacking -- Stacking 集成训练与评估
    generate_comparison_table -- 返回算法核心假设与适用条件对比表
    plot_radar_chart -- 多维度雷达图模型对比
    plot_parameter_curves -- 手动调参参数-性能曲线
"""
import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PROCESSED_TRAIN_PATH, MODEL_DIR, FIGURES_DIR,
                        ACTIVE_MODEL, MODEL_HYPERPARAMS, RANDOM_SEED,
                        AVAILABLE_MODELS, DEFAULT_CV_FOLDS)
from src.models import UsedCarModelFactory

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _make_stratified_folds(y, n_splits=5, n_bins=10, random_state=42):
    """按价格分位数创建分层 KFold 划分。

    将价格分为 n_bins 个分位数区间，确保每折的价格分布一致，
    避免某折全是低价车或全是高价车的极端情况。

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y_binned))


class ModelTrainer:
    """统一模型训练器。

    封装模型实例化、训练、预测、交叉验证与持久化的全生命周期。
    训练时目标变量在对数空间 (log1p)，预测时自动还原到真实金额空间。

    Parameters
    ----------
    model_name : str
        模型名称
    params : dict or None
        覆盖默认超参数的字典
    random_state : int
    use_log_target : bool
        是否对目标做 log1p 变换 (default=True)
    """

    def __init__(self, model_name, params=None, random_state=42,
                 use_log_target=True):
        self.model_name = model_name.lower().strip()
        self.use_log_target = use_log_target
        self.random_state = random_state
        merged = MODEL_HYPERPARAMS.get(self.model_name, {}).copy()
        if params:
            merged.update(params)
        self.params = merged
        self.model_ = None

    def train(self, X, y):
        """在完整训练集上拟合模型。y 应为原始价格（自动 log1p）。"""
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = np.asarray(y)
        if self.use_log_target:
            y_arr = np.log1p(y_arr)
        self.model_ = UsedCarModelFactory.create_model(
            self.model_name, **self.params)
        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X, return_real=True):
        """预测。return_real=True 时还原到真实金额空间 (expm1)。"""
        if self.model_ is None:
            raise RuntimeError("模型尚未训练，请先调用 train()")
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        preds = self.model_.predict(X_arr)
        if self.use_log_target and return_real:
            preds = np.expm1(preds)
        return preds

    def cross_validate(self, X, y, n_splits=5, stratified=True):
        """K 折交叉验证（支持分层），返回各折指标 dict。

        Parameters
        ----------
        stratified : bool
            是否使用分层 KFold（按价格分位数分层，推荐 True）
        """
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = np.asarray(y)
        metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'mape': []}

        if stratified:
            folds = _make_stratified_folds(y_arr, n_splits=n_splits,
                                           random_state=self.random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            folds = list(kf.split(X_arr))

        for train_idx, val_idx in folds:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            y_tr_log = np.log1p(y_tr)
            model = UsedCarModelFactory.create_model(
                self.model_name, **self.params)
            model.fit(X_tr, y_tr_log)
            preds_log = model.predict(X_val)
            preds_real = np.expm1(preds_log)
            preds_real = np.maximum(preds_real, 0)

            metrics['mae'].append(mean_absolute_error(y_val, preds_real))
            metrics['mse'].append(mean_squared_error(y_val, preds_real))
            metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))
            metrics['r2'].append(r2_score(np.log1p(y_val), preds_log))
            metrics['mape'].append(
                np.mean(np.abs((y_val - preds_real) / np.maximum(y_val, 1))) * 100)
        return metrics

    def save_model(self, filepath=None):
        """保存模型到 models/ 目录。"""
        if self.model_ is None:
            raise RuntimeError("模型尚未训练，无法保存")
        os.makedirs(MODEL_DIR, exist_ok=True)
        if filepath is None:
            filepath = os.path.join(
                MODEL_DIR, f"{self.model_name}_model.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_, f)
        return filepath

    @staticmethod
    def load_model(filepath):
        """从 pickle 文件加载模型。"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_feature_importance(self, feature_names):
        """获取特征重要性（树模型专有）；线性模型返回系数。"""
        if self.model_ is None:
            return None
        if hasattr(self.model_, 'feature_importances_'):
            imp = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            imp = np.abs(self.model_.coef_).ravel()
        else:
            return None
        return pd.DataFrame({
            'feature': feature_names[:len(imp)],
            'importance': imp
        }).sort_values('importance', ascending=False)


def train_all_models(X, y, models=None, cv_folds=5, stratified=True):
    """同一 CV 拆分下训练所有模型，返回比较 DataFrame。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : array-like
        原始价格
    models : list[str] or None
        模型列表，默认取 AVAILABLE_MODELS
    cv_folds : int
    stratified : bool
        是否使用分层 KFold（推荐，确保各折价格分布一致）

    Returns
    -------
    pd.DataFrame
        列: model, R2_mean, R2_std, MAE_mean, MAE_std,
             RMSE_mean, RMSE_std, MAPE_mean, MAPE_std, train_time_s
    """
    if models is None:
        models = AVAILABLE_MODELS[:]

    X_arr = X.values.astype(float) if hasattr(X, 'values') else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # 同一分层 CV 拆分保证公平对比
    if stratified:
        folds = _make_stratified_folds(y_arr, n_splits=cv_folds, random_state=RANDOM_SEED)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        folds = list(kf.split(X_arr))

    results = []
    for model_name in models:
        t0 = time.time()
        fold_mae, fold_rmse, fold_r2, fold_mape = [], [], [], []

        for train_idx, val_idx in folds:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            trainer = ModelTrainer(model_name)
            trainer.train(pd.DataFrame(X_tr, columns=X.columns), y_tr)
            preds_real = trainer.predict(pd.DataFrame(X_val, columns=X.columns))
            preds_real = np.maximum(preds_real, 0)

            mae = mean_absolute_error(y_val, preds_real)
            rmse = np.sqrt(mean_squared_error(y_val, preds_real))
            r2 = r2_score(np.log1p(y_val), np.log1p(preds_real))
            mape = np.mean(np.abs((y_val - preds_real) / np.maximum(y_val, 1))) * 100

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)
            fold_mape.append(mape)

        elapsed = time.time() - t0
        results.append({
            'model': model_name.upper().replace('_', ' '),
            'R2_mean': np.mean(fold_r2), 'R2_std': np.std(fold_r2),
            'MAE_mean': np.mean(fold_mae), 'MAE_std': np.std(fold_mae),
            'RMSE_mean': np.mean(fold_rmse), 'RMSE_std': np.std(fold_rmse),
            'MAPE_mean': np.mean(fold_mape), 'MAPE_std': np.std(fold_mape),
            'train_time_s': elapsed
        })

    return pd.DataFrame(results).sort_values('R2_mean', ascending=False)


def train_stacking(X, y, cv_folds=5, stratified=True):
    """训练并评估 Stacking 集成模型。

    使用分层 KFold 交叉验证，每个基模型使用调优后的参数。
    元学习器为 ElasticNet (L1+L2)，passthrough=True。

    Returns
    -------
    dict : 含 R2_mean, MAE_mean, RMSE_mean, MAPE_mean 及标准差
    """
    X_arr = X.values.astype(float) if hasattr(X, 'values') else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if stratified:
        folds = _make_stratified_folds(y_arr, n_splits=cv_folds, random_state=RANDOM_SEED)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        folds = list(kf.split(X_arr))

    fold_mae, fold_rmse, fold_r2, fold_mape = [], [], [], []

    print(f"\n  [Stacking] 训练中 ({cv_folds}-fold CV, 分层={stratified})...")
    for fold_i, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
        y_tr_log = np.log1p(y_tr)

        model = UsedCarModelFactory.create_model('stacking')
        model.fit(X_tr, y_tr_log)
        preds_log = model.predict(X_val)
        preds_real = np.expm1(preds_log)
        preds_real = np.maximum(preds_real, 0)

        mae = mean_absolute_error(y_val, preds_real)
        rmse = np.sqrt(mean_squared_error(y_val, preds_real))
        r2 = r2_score(np.log1p(y_val), preds_log)
        mape = np.mean(np.abs((y_val - preds_real) / np.maximum(y_val, 1))) * 100

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_mape.append(mape)
        print(f"    Fold {fold_i}: R2={r2:.4f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}, MAPE={mape:.1f}%")

    return {
        'R2_mean': np.mean(fold_r2), 'R2_std': np.std(fold_r2),
        'MAE_mean': np.mean(fold_mae), 'MAE_std': np.std(fold_mae),
        'RMSE_mean': np.mean(fold_rmse), 'RMSE_std': np.std(fold_rmse),
        'MAPE_mean': np.mean(fold_mape), 'MAPE_std': np.std(fold_mape),
    }


def generate_comparison_table():
    """返回算法核心假设与适用条件对比表 (6模型)。"""
    table = [
        {
            '算法': 'Ridge (岭回归)',
            '模型类型': '线性模型 (L2 正则化)',
            '核心假设': '特征与目标线性相关；误差 i.i.d. 同方差',
            '优势': '训练极快、可解释性强、不易过拟合',
            '劣势': '无法捕获非线性；对异常值敏感',
            '适用场景': '基线模型；需要强可解释性的业务场景'
        },
        {
            '算法': 'ElasticNet',
            '模型类型': '线性模型 (L1+L2 正则化)',
            '核心假设': '同 Ridge；L1 可做特征选择',
            '优势': '兼具 Lasso 特征选择与 Ridge 稳定性；自动筛选关键特征',
            '劣势': '需调两个超参数(alpha, l1_ratio)；仍为线性模型',
            '适用场景': '特征维度高但多数特征可能冗余的场景'
        },
        {
            '算法': 'Random Forest (随机森林)',
            '模型类型': '集成树模型 (Bagging)',
            '核心假设': '无严格分布假设；特征与目标可存在复杂非线性',
            '优势': '对异常值和缺失值鲁棒；自动捕获特征交互',
            '劣势': '训练和推理较慢；模型体积大',
            '适用场景': '特征间存在复杂交互；无需过多调参即可获得强基线'
        },
        {
            '算法': 'LightGBM',
            '模型类型': '梯度提升树 (Boosting, 直方图)',
            '核心假设': '无严格分布假设；leaf-wise 生长效率高',
            '优势': '训练极快；内存效率高；精度通常最高',
            '劣势': '小数据易过拟合；leaf-wise 可能不稳定',
            '适用场景': '大规模数据集；追求最高精度的生产环境'
        },
        {
            '算法': 'KNN (K近邻)',
            '模型类型': '基于距离的实例学习',
            '核心假设': '相似车况的二手车具有相似价格',
            '优势': '完全非线性；无需训练；直观可解释',
            '劣势': '推理慢(O(n)复杂度)；高维数据距离失效；对量纲敏感',
            '适用场景': '局部价格参考；与树/线性模型互补的集成基模型'
        },
        {
            '算法': 'Stacking (集成融合)',
            '模型类型': '异质集成 (Ridge+EN+RF+LGB+KNN → ElasticNet)',
            '核心假设': '不同范式的算法误差不相关，可互补',
            '优势': '融合线性/Bagging/Boosting/距离四种范式；稳健性最强',
            '劣势': '训练时间长；可解释性差',
            '适用场景': '追求极致精度与稳健性的最终模型'
        }
    ]
    return pd.DataFrame(table)


def plot_radar_chart(results_df, save_path=None):
    """绘制多维度雷达图模型对比（含 Stacking）。

    维度: R2, 1/MAE, 1/RMSE, 1/MAPE, 1/Train Time
    """
    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, 'model_radar_comparison.png')

    df = results_df.copy()
    n_models = len(df)

    scores = {}
    scores['R2'] = np.maximum(df['R2_mean'].values, 0)
    scores['1/MAE'] = 1.0 / (df['MAE_mean'].values + 1e-8)
    scores['1/RMSE'] = 1.0 / (df['RMSE_mean'].values + 1e-8)
    scores['1/MAPE'] = 1.0 / (df['MAPE_mean'].values + 1e-8)
    scores['1/训练时间'] = 1.0 / (df['train_time_s'].values + 1e-8)

    dims = list(scores.keys())
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['#2F5496', '#C00000', '#ED7D31', '#70AD47', '#7030A0',
              '#00B0F0', '#FFC000', '#264653']

    for i, (name, vals) in enumerate(scores.items()):
        vals_norm = vals / (vals.max() + 1e-8)
        scores[name] = vals_norm

    for i in range(n_models):
        values = [scores[d][i] for d in dims]
        values += values[:1]
        ax.fill(angles, values, alpha=0.08, color=colors[i % len(colors)])
        ax.plot(angles, values, 'o-', linewidth=2.2, color=colors[i % len(colors)],
                label=df.iloc[i]['model'], markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.set_title('多模型多维雷达图对比\n(全部维度归一化至 [0,1], 越靠外越优)',
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_parameter_curves(model_name, param_configs, X, y, cv_folds=5,
                          save_dir=None, stratified=True):
    """对指定模型的多个参数绘制参数-性能曲线。

    Parameters
    ----------
    model_name : str
    param_configs : list of (param_name, param_values)
    X : pd.DataFrame
    y : array-like (原始价格)
    cv_folds : int
    save_dir : str
    stratified : bool
    """
    if save_dir is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_dir = FIGURES_DIR

    X_arr = X.values.astype(float) if hasattr(X, 'values') else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    saved = {}
    for param_name, param_values in param_configs:
        print(f"\n  [Tuning] {model_name.upper()} -- {param_name}: {param_values}")
        train_r2_means, val_r2_means = [], []

        for val in param_values:
            fold_train_r2, fold_val_r2 = [], []

            if stratified:
                folds = _make_stratified_folds(y_arr, n_splits=cv_folds, random_state=RANDOM_SEED)
            else:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
                folds = list(kf.split(X_arr))

            for train_idx, val_idx in folds:
                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
                y_tr_log = np.log1p(y_tr)

                model = UsedCarModelFactory.create_model(
                    model_name, **{param_name: val})
                model.fit(X_tr, y_tr_log)
                preds_tr = model.predict(X_tr)
                preds_val = model.predict(X_val)

                fold_train_r2.append(r2_score(y_tr_log, preds_tr))
                fold_val_r2.append(r2_score(np.log1p(y_val), preds_val))

            train_r2_means.append(np.mean(fold_train_r2))
            val_r2_means.append(np.mean(fold_val_r2))
            try:
                print("    %s=%-6s -> Train R2=%.4f  Val R2=%.4f" % (
                    param_name, str(val), train_r2_means[-1], val_r2_means[-1]))
            except UnicodeEncodeError:
                print(("    %s=%-6s -> Train R2=%.4f  Val R2=%.4f" % (
                    param_name, str(val), train_r2_means[-1], val_r2_means[-1])).encode('ascii', errors='replace').decode('ascii'))

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_values, train_r2_means, 'o-', color='#C00000',
                linewidth=2, markersize=8, label='Train R2')
        ax.plot(param_values, val_r2_means, 's--', color='#2F5496',
                linewidth=2, markersize=8, label='Validation R2')

        best_idx = np.argmax(val_r2_means)
        ax.annotate(f'Best: {param_values[best_idx]}\nVal R2={val_r2_means[best_idx]:.4f}',
                    xy=(param_values[best_idx], val_r2_means[best_idx]),
                    xytext=(param_values[best_idx], val_r2_means[best_idx] - 0.03),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='green'))

        ax.set_xlabel(f'Parameter: {param_name}', fontsize=12)
        ax.set_ylabel('R2 Score', fontsize=12)
        ax.set_title(f'{model_name.upper()} — {param_name} Parameter-Performance Curve\n'
                     f'({cv_folds}-fold Stratified CV)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

        path = os.path.join(save_dir, f'param_{model_name}_{param_name}_curve.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        saved[param_name] = path
        print(f"  [OK] Chart saved: {path}")

    return saved


def plot_predictions_vs_actual(model_name, X, y, cv_folds=5, save_path=None,
                                sample_size=5000, stratified=True):
    """绘制预测值 vs 真实值散点图 + 残差分布直方图。

    使用 OOF (Out-of-Fold) 预测避免数据泄露，确保评估客观。

    Parameters
    ----------
    model_name : str
        模型名称 (如 'lightgbm')
    X : pd.DataFrame
    y : array-like (原始价格)
    cv_folds : int
    save_path : str or None
    sample_size : int
        散点图采样数（数据量大时全画会重叠严重）
    stratified : bool
    """
    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, 'validation_predictions_vs_actual.png')

    X_arr = X.values.astype(float) if hasattr(X, 'values') else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # OOF 预测
    if stratified:
        folds = _make_stratified_folds(y_arr, n_splits=cv_folds, random_state=RANDOM_SEED)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        folds = list(kf.split(X_arr))

    oof_preds = np.zeros(len(y_arr))
    for train_idx, val_idx in folds:
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr = y_arr[train_idx]
        model = UsedCarModelFactory.create_model(model_name)
        model.fit(X_tr, np.log1p(y_tr))
        oof_preds[val_idx] = np.expm1(model.predict(X_val))
    oof_preds = np.maximum(oof_preds, 0)

    residuals = y_arr - oof_preds
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / np.maximum(y_arr, 1))) * 100
    r2 = r2_score(np.log1p(y_arr), np.log1p(oof_preds))

    # ---- 采样散点图 ----
    n = min(sample_size, len(y_arr))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(y_arr), size=n, replace=False)
    y_sample, pred_sample, res_sample = y_arr[idx], oof_preds[idx], residuals[idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # -- 左图: 预测 vs 真实散点图 --
    ax = axes[0]
    max_val = max(y_sample.max(), pred_sample.max()) * 1.05
    ax.scatter(y_sample, pred_sample, alpha=0.25, s=8, c='#2F5496',
               edgecolors='none', label=f'n={n:,} samples')
    ax.plot([0, max_val], [0, max_val], '--', color='#C00000', linewidth=2,
            label='y=x (完美预测)')
    ax.fill_between([0, max_val], [0, max_val],
                    [0, max_val * 1.15], alpha=0.06, color='red', label='高估区域')
    ax.fill_between([0, max_val], [0, max_val * 0.85],
                    [0, max_val], alpha=0.06, color='blue', label='低估区域')
    ax.set_xlabel('True Price (CNY)', fontsize=12)
    ax.set_ylabel('Predicted Price (CNY)', fontsize=12)
    ax.set_title(f'{model_name.upper()} — Predicted vs Actual\n'
                 f'R²={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # -- 右图: 残差分布直方图 --
    ax = axes[1]
    # 剔除极端异常值使直方图可读
    q_low, q_high = np.percentile(residuals, [1, 99])
    res_trimmed = residuals[(residuals >= q_low) & (residuals <= q_high)]
    ax.hist(res_trimmed, bins=80, color='#2F5496', edgecolor='white',
            alpha=0.85, density=True)
    ax.axvline(x=0, color='#C00000', linewidth=2, linestyle='--', label='残差=0 (完美)')
    ax.axvline(x=np.mean(residuals), color='#ED7D31', linewidth=2, linestyle='-',
               label=f'均值={np.mean(residuals):,.0f}')
    ax.set_xlabel('Residual = True − Predicted (CNY)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Residual Distribution ({model_name.upper()})\n'
                 f'均值={np.mean(residuals):,.0f}  标准差={np.std(residuals):,.0f}  '
                 f'偏度={pd.Series(residuals).skew():.2f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

    # 文字分析
    skew_val = pd.Series(residuals).skew()
    if abs(skew_val) < 0.3:
        skew_note = '残差近似对称，模型无明显系统性偏差'
    elif skew_val > 0:
        skew_note = '残差右偏(正残差多)→ 模型倾向低估真实价格，对高价车保守'
    else:
        skew_note = '残差左偏(负残差多)→ 模型倾向高估真实价格，需检查特征覆盖度'

    fig.text(0.5, -0.02,
             f'[解读] {skew_note}。散点越靠近y=x线预测越准；'
             f'高价位区间(>5万)散点离散度增大→高价车预测不确定性更高。',
             ha='center', fontsize=10, style='italic', color='#555555',
             transform=fig.transFigure)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 预测 vs 真实图: {save_path}")
    print(f"  [解读] {skew_note}")
    return save_path
