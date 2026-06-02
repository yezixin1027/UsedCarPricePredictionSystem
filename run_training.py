"""
二手车价格预测系统 · 模型训练与评估入口
=========================================

默认训练全部 5 个模型 + Stacking 集成，使用分层 5 折 CV。
所有模型训练后自动保存权重到 models/ 目录。

Usage:
    python run_training.py                        # 全模型 + Stacking（默认）
    python run_training.py --model lightgbm       # 仅训练指定模型
    python run_training.py --no-stacking          # 跳过 Stacking
    python run_training.py --rerun-pipeline       # 先重跑数据流水线
    python run_training.py --predict              # 生成测试集预测
    python run_training.py --tune                 # 运行超参数调优
"""
import os, sys, time, pickle, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import (PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH,
                        MODEL_DIR, FIGURES_DIR,
                        AVAILABLE_MODELS, RANDOM_SEED, DEFAULT_CV_FOLDS)
from src.train import (ModelTrainer, train_all_models, train_stacking,
                       generate_comparison_table, plot_radar_chart)
from src.models import UsedCarModelFactory

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def parse_args():
    p = argparse.ArgumentParser(description='二手车价格预测 — 五模型训练与 Stacking')
    p.add_argument('--model', type=str, default=None,
                   help=f'模型名称: {", ".join(AVAILABLE_MODELS)}')
    p.add_argument('--no-stacking', action='store_true', help='跳过 Stacking')
    p.add_argument('--rerun-pipeline', action='store_true', help='重跑数据流水线')
    p.add_argument('--cv-folds', type=int, default=DEFAULT_CV_FOLDS, help=f'CV 折数')
    p.add_argument('--predict', action='store_true', help='生成测试集预测')
    p.add_argument('--tune', action='store_true', help='运行超参数调优')
    p.add_argument('--tune-trials', type=int, default=30, help='Optuna 试验次数')
    return p.parse_args()


# ================================================================
# 阶段 1: 数据加载
# ================================================================
def stage_load_data(rerun=False):
    safe_print("=" * 70)
    safe_print("[Stage 1] Load Feature Matrix")
    safe_print("=" * 70)
    if rerun:
        safe_print("  -> Re-running data pipeline...")
        from run_pipeline import run_pipeline
        run_pipeline()
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_TRAIN_PATH}")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    y = np.expm1(train_df['log_price'].values)
    safe_print(f"  Features: {X.shape[0]:,} samples x {X.shape[1]} features")
    safe_print(f"  Price range: [{y.min():.0f}, {y.max():.0f}]")
    safe_print(f"  Columns: {list(X.columns)}")
    return X, y


# ================================================================
# 阶段 2: 五模型训练对比
# ================================================================
def stage_train_compare(X, y, args):
    safe_print("\n" + "=" * 70)
    safe_print("[Stage 2] 5-Model Training (Stratified KFold CV)")
    safe_print("=" * 70)
    models = [args.model.lower()] if args.model else AVAILABLE_MODELS[:]
    safe_print(f"  Models ({len(models)}): {[m.upper() for m in models]}")
    results = train_all_models(X, y, models=models, cv_folds=args.cv_folds)
    safe_print(f"\n  {'Model':<18} {'R2':>8} {'+/-':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    safe_print(f"  {'-' * 72}")
    for _, row in results.iterrows():
        safe_print(f"  {row['model']:<18} {row['R2_mean']:>8.4f} {row['R2_std']:>8.4f} "
                   f"{row['MAE_mean']:>12,.0f} {row['RMSE_mean']:>12,.0f} {row['MAPE_mean']:>7.1f}%")
    best = results.iloc[0]
    safe_print(f"\n  [Best Single] {best['model']} — R2={best['R2_mean']:.4f}, "
               f"MAE={best['MAE_mean']:,.0f}")
    return results


# ================================================================
# 阶段 3: Stacking 集成
# ================================================================
def stage_stacking(X, y, results_df, args):
    safe_print("\n" + "=" * 70)
    safe_print("[Stage 3] Stacking Ensemble (LGB + XGB + CatBoost -> ElasticNet)")
    safe_print("=" * 70)
    stacking = train_stacking(X, y, cv_folds=args.cv_folds)
    safe_print(f"\n  Stacking R2={stacking['R2_mean']:.4f} +/- {stacking['R2_std']:.4f}")
    safe_print(f"  MAE={stacking['MAE_mean']:,.0f}, RMSE={stacking['RMSE_mean']:,.0f}, "
               f"MAPE={stacking['MAPE_mean']:.1f}%")
    best_single = results_df.iloc[0]
    r2_gain = stacking['R2_mean'] - best_single['R2_mean']
    safe_print(f"  vs Best Single ({best_single['model']}): R2 {r2_gain:+.4f}")
    return stacking


# ================================================================
# 阶段 4: 保存所有模型权重
# ================================================================
def stage_save_all_models(X, y, results_df, stacking_metrics):
    """训练并保存全部 5 个模型的最终权重到 models/ 目录。

    每个模型在全部训练数据上训练（非 CV），保存为 .pkl 文件。
    同时保存特征重要性报告。
    """
    safe_print("\n" + "=" * 70)
    safe_print("[Stage 4] Save All Model Weights to models/")
    safe_print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    saved = []
    all_importances = {}

    # 4a. 保存全部 5 个独立模型
    for _, row in results_df.iterrows():
        model_name = row['model'].lower().replace(' ', '_')
        safe_print(f"\n  Training {model_name.upper()} on full dataset...")

        trainer = ModelTrainer(model_name)
        trainer.train(X, y)
        filepath = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        trainer.save_model(filepath)
        saved.append(filepath)

        # 提取特征重要性
        imp = trainer.get_feature_importance(list(X.columns))
        if imp is not None:
            all_importances[model_name] = imp
            safe_print(f"    Top-3 features: {', '.join(imp['feature'].head(3).values)}")

        safe_print(f"    Saved: {filepath}")

    # 4b. 保存 Stacking 模型
    if stacking_metrics is not None:
        safe_print(f"\n  Training STACKING on full dataset...")
        stacking_model = UsedCarModelFactory.create_model('stacking')
        stacking_model.fit(X.values.astype(float), np.log1p(y))
        filepath = os.path.join(MODEL_DIR, "stacking_model.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(stacking_model, f)
        saved.append(filepath)
        safe_print(f"    Saved: {filepath}")

    # 4c. 保存特征重要性汇总表
    imp_summary = []
    for model_name, imp_df in all_importances.items():
        for _, row in imp_df.head(5).iterrows():
            imp_summary.append({
                'model': model_name.upper(),
                'feature': row['feature'],
                'importance': row['importance']
            })

    if imp_summary:
        imp_summary_df = pd.DataFrame(imp_summary)
        imp_path = os.path.join(MODEL_DIR, "feature_importance_summary.csv")
        imp_summary_df.to_csv(imp_path, index=False)
        safe_print(f"\n  Feature importance summary: {imp_path}")

    # 4d. 保存模型性能对比表
    results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
    if stacking_metrics:
        stacking_df = pd.DataFrame([{
            'model': 'STACKING',
            'R2_mean': stacking_metrics['R2_mean'],
            'R2_std': stacking_metrics['R2_std'],
            'MAE_mean': stacking_metrics['MAE_mean'],
            'MAE_std': stacking_metrics['MAE_std'],
            'RMSE_mean': stacking_metrics['RMSE_mean'],
            'RMSE_std': stacking_metrics['RMSE_std'],
            'MAPE_mean': stacking_metrics['MAPE_mean'],
            'MAPE_std': stacking_metrics['MAPE_std'],
            'train_time_s': 0
        }])
        stacking_df.to_csv(os.path.join(MODEL_DIR, "stacking_result.csv"), index=False)

    safe_print(f"\n  {'=' * 60}")
    safe_print(f"  Total models saved: {len(saved)}")
    safe_print(f"  Model directory: {MODEL_DIR}")
    for s in saved:
        safe_print(f"    {os.path.basename(s)}")
    safe_print(f"  {'=' * 60}")
    return saved


# ================================================================
# 阶段 5: 测试集预测
# ================================================================
def stage_test_predict(X, y):
    safe_print(f"\n[Stage 5] Generating test set predictions...")
    if not os.path.exists(PROCESSED_TEST_PATH):
        safe_print("  Test features not found, skipping")
        return
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    model_path = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        preds = np.expm1(model.predict(test_df.values.astype(float)))
        preds = np.maximum(preds, 0)
        out_path = os.path.join(os.path.dirname(PROCESSED_TEST_PATH),
                                '..', 'reports', 'test_predictions.csv')
        output = pd.DataFrame({'predicted_price': preds})
        output.to_csv(os.path.abspath(out_path), index=False)
        safe_print(f"  Saved: {os.path.abspath(out_path)}")
        safe_print(f"  Price range: [{preds.min():,.0f}, {preds.max():,.0f}]")


# ================================================================
# 主入口
# ================================================================
def run_training():
    start_time = time.time()
    args = parse_args()

    safe_print("=" * 70)
    safe_print("  Used Car Price Prediction System")
    safe_print("=" * 70)
    mode = f"Single ({args.model.upper()})" if args.model else "All 5 Models + Stacking"
    safe_print(f"  Mode: {mode} | CV: {args.cv_folds}-fold Stratified KFold")
    safe_print("=" * 70)

    # 可选: 超参数调优
    if args.tune:
        from src.tune_all_models import run_full_tuning
        run_full_tuning(n_trials=args.tune_trials)

    # 阶段 1: 数据加载
    X, y = stage_load_data(rerun=args.rerun_pipeline)

    # 阶段 2: 五模型训练对比
    results = stage_train_compare(X, y, args)

    # 阶段 3: Stacking
    stacking = None
    if not args.no_stacking and not args.model:
        stacking = stage_stacking(X, y, results, args)

    # 阶段 4: 保存全部模型权重
    stage_save_all_models(X, y, results, stacking)

    # 算法对比表
    safe_print("\n[Algorithm Comparison]")
    for _, row in generate_comparison_table().iterrows():
        safe_print(f"  {row['算法']}: {row['优势'][:80]}...")

    # 阶段 5: 测试集预测
    if args.predict:
        stage_test_predict(X, y)

    elapsed = time.time() - start_time
    safe_print(f"\n{'=' * 70}")
    safe_print(f"  Training Complete! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    safe_print(f"  Models: {MODEL_DIR} | Figures: {FIGURES_DIR}")
    safe_print(f"{'=' * 70}")


if __name__ == "__main__":
    run_training()
