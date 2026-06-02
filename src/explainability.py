# src/explainability.py
"""
模型可解释性分析模块 (SHAP)
===========================

基于 SHAP (SHapley Additive exPlanations) 对最佳模型进行:
1. 全局特征重要性 (Summary Plot)
2. 特征依赖图 (Dependence Plot) — 关键特征的边际效应
3. 单样本预测分解 (Waterfall Plot) — 解释单个预测的定价逻辑

Usage:
    python src/explainability.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH,
                        MODEL_DIR, FIGURES_DIR, RANDOM_SEED)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_shap_analysis(model_path=None, n_samples=5000):
    """运行 SHAP 可解释性分析，生成 4 张解释性图表。

    Parameters
    ----------
    model_path : str or None
        模型 pickle 路径，默认使用 models/lightgbm_model.pkl
    n_samples : int
        SHAP 背景样本数（过多会很慢）
    """
    print("=" * 60)
    print("[SHAP 可解释性分析]")
    print("=" * 60)

    # 1. 加载数据
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    feature_names = list(X.columns)

    # 2. 加载模型
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型不存在: {model_path}")
            print("  请先运行: python run_training.py --model lightgbm")
            return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  模型: {model_path}")
    print(f"  数据: {X.shape[0]:,} 样本 x {X.shape[1]} 特征")

    # 3. SHAP 分析
    try:
        import shap
        shap.initjs()
    except ImportError:
        print("[INFO] 正在安装 shap...")
        os.system(f"{sys.executable} -m pip install shap -q")
        import shap
        shap.initjs()

    # 采样背景数据
    bg_idx = np.random.RandomState(RANDOM_SEED).choice(
        len(X), size=min(n_samples, len(X)), replace=False)
    X_bg = X.iloc[bg_idx].values.astype(float)

    # 创建 explainer (TreeExplainer 对树模型最快)
    print("  创建 SHAP TreeExplainer...")
    if hasattr(model, 'objective'):
        explainer = shap.TreeExplainer(model, X_bg)
    else:
        # 对于非树模型使用 KernelExplainer 的简化版本
        explainer = shap.KernelExplainer(
            model.predict, X_bg[:100], feature_names=feature_names)

    # 计算 SHAP 值 (使用部分样本加速)
    X_sample = X.iloc[:min(2000, len(X))].values.astype(float)
    print(f"  计算 SHAP 值 (样本={len(X_sample)})...")
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 回归任务取第一个

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- 图 1: Summary Plot (特征重要性) ----
    print("  [1/4] 生成 Summary Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

    # ---- 图 2: Feature Importance (Bar) ----
    print("  [2/4] 生成 Feature Importance 条形图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_importance_bar.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

    # ---- 图 3: Dependence Plot (关键特征) ----
    print("  [3/4] 生成 Dependence Plots...")
    top_features = np.argsort(np.abs(shap_values).mean(0))[-4:][::-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feat_idx in enumerate(top_features):
        shap.dependence_plot(feat_idx, shap_values, X_sample,
                             feature_names=feature_names,
                             show=False, ax=axes[i])
        axes[i].set_title(f'{feature_names[feat_idx]} SHAP Dependence',
                          fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_dependence.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

    # ---- 图 4: Waterfall (单样本解释) ----
    print("  [4/4] 生成 Waterfall 单样本解释...")
    sample_idx = 0
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[sample_idx],
                         base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                         else explainer.expected_value[0],
                         data=X_sample[sample_idx],
                         feature_names=feature_names),
        show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

    # 5. 输出特征重要性排序
    mean_abs_shap = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\n  SHAP 特征重要性 Top-10:")
    print(f"  {'─' * 50}")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:.6f}")

    print(f"\n  [OK] SHAP 图表已保存到: {FIGURES_DIR}/")
    print(f"    - shap_summary.png")
    print(f"    - shap_importance_bar.png")
    print(f"    - shap_dependence.png")
    print(f"    - shap_waterfall.png")

    return importance_df


if __name__ == "__main__":
    run_shap_analysis()
