# modeling/run_modeling.py
# ========================
# 模型构建与评估 — 阶段入口
#
# 按顺序执行:
#   s1: 5模型训练 (4.1-4.2)
#   s2: 超参数调优 (4.2, 可选)
#   s3: 模型性能对比 (4.3)
#   s4: 最佳模型深度分析 — SHAP+审计 (4.4)
import os, sys, time, pickle, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_DIR, FIGURES_DIR,
                    AVAILABLE_MODELS, RANDOM_SEED, DEFAULT_CV_FOLDS)
from src.train import (ModelTrainer, train_all_models, train_stacking,
                       generate_comparison_table, plot_radar_chart)
from src.models import UsedCarModelFactory

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_modeling(skip_tune=True, skip_stacking=False, cv_folds=5):
    """执行全部模型训练与评估流程"""
    start_time = time.time()
    safe_print("=" * 70)
    safe_print("  模型构建与评估")
    safe_print(f"  模式: {'调优模式' if not skip_tune else '默认参数模式'} | CV: {cv_folds}-fold")
    safe_print("=" * 70)

    # ---- 加载数据 ----
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        safe_print("[ERROR] 预处理数据不存在, 请先运行 preprocessing/run_preprocessing.py")
        return
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    y = np.expm1(train_df['log_price'].values)
    safe_print(f"\n  特征矩阵: {X.shape[0]:,} x {X.shape[1]}")

    # ---- s1: 5模型训练 (4.1-4.2) ----
    safe_print(f"\n[4.1-4.2] 五模型训练 ({cv_folds}-fold Stratified CV)")
    results = train_all_models(X, y, models=AVAILABLE_MODELS, cv_folds=cv_folds)
    safe_print(f"\n  {'Model':<18} {'R2':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    safe_print(f"  {'─' * 65}")
    for _, row in results.iterrows():
        safe_print(f"  {row['model']:<18} {row['R2_mean']:>8.4f} {row['MAE_mean']:>12,.0f} "
                   f"{row['RMSE_mean']:>12,.0f} {row['MAPE_mean']:>7.1f}%")
    best = results.iloc[0]
    safe_print(f"\n  [Best] {best['model']} R2={best['R2_mean']:.4f} MAE={best['MAE_mean']:,.0f}")

    # ---- s2: 超参数调优 (可选) ----
    if not skip_tune:
        safe_print(f"\n[4.2] 超参数调优")
        from src.tune_all_models import run_full_tuning
        run_full_tuning(n_trials=30)

    # ---- s3: Stacking + 雷达图 (4.3) ----
    stacking = None
    if not skip_stacking:
        safe_print(f"\n[4.3] Stacking 集成 (LGB+XGB+Cat -> ElasticNet)")
        stacking = train_stacking(X, y, cv_folds=cv_folds)
        safe_print(f"  Stacking R2={stacking['R2_mean']:.4f} MAE={stacking['MAE_mean']:,.0f}")

    # 雷达图
    if stacking:
        stack_row = pd.DataFrame([{'model': 'STACKING', 'R2_mean': stacking['R2_mean'],
                                    'MAE_mean': stacking['MAE_mean'], 'RMSE_mean': stacking['RMSE_mean'],
                                    'MAPE_mean': stacking['MAPE_mean'], 'train_time_s': 0}])
        plot_df = pd.concat([results, stack_row], ignore_index=True)
    else:
        plot_df = results
    radar_path = plot_radar_chart(plot_df)
    safe_print(f"  [OK] 雷达图: {radar_path}")

    # 算法对比表
    safe_print(f"\n[4.3] 算法核心假设与适用条件对比:")
    for _, row in generate_comparison_table().iterrows():
        safe_print(f"  {row['算法']}: {row['优势'][:100]}...")

    # ---- s4: 保存模型权重 + SHAP (4.4) ----
    safe_print(f"\n[4.4] 保存模型权重 & 特征重要性")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for _, row in results.iterrows():
        model_name = row['model'].lower().replace(' ', '_')
        trainer = ModelTrainer(model_name)
        trainer.train(X, y)
        trainer.save_model(os.path.join(MODEL_DIR, f"{model_name}_model.pkl"))
        safe_print(f"  [OK] {model_name}_model.pkl")

    if stacking:
        stacking_model = UsedCarModelFactory.create_model('stacking')
        stacking_model.fit(X.values.astype(float), np.log1p(y))
        with open(os.path.join(MODEL_DIR, "stacking_model.pkl"), 'wb') as f:
            pickle.dump(stacking_model, f)
        safe_print(f"  [OK] stacking_model.pkl")

    # 保存对比表
    results.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

    elapsed = time.time() - start_time
    safe_print(f"\n{'=' * 60}")
    safe_print(f"  第4章完成 | 耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    safe_print(f"  模型目录: {MODEL_DIR}")
    safe_print(f"{'=' * 60}")
    return results, stacking


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--tune', action='store_true', help='运行超参数调优')
    p.add_argument('--no-stacking', action='store_true', help='跳过Stacking')
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()
    run_modeling(skip_tune=not args.tune, skip_stacking=args.no_stacking, cv_folds=args.cv)
