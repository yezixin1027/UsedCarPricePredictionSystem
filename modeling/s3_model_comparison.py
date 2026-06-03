# modeling/s3_model_comparison.py
# ================================
# 论文 4.3 模型性能对比分析
#
# 多维度对比 (雷达图): R2, 1/MAE, 1/RMSE, 1/MAPE, 1/训练时间
# 算法对比表: 核心假设、优势、劣势、适用场景
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH, MODEL_DIR, FIGURES_DIR
from src.train import train_all_models, train_stacking, plot_radar_chart, generate_comparison_table

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_model_comparison():
    """运行 4.3 模型性能对比分析"""
    safe_print("=" * 60)
    safe_print("  4.3 模型性能对比分析")
    safe_print("=" * 60)

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    X = train_df.drop(columns=['log_price'])
    y = np.expm1(train_df['log_price'].values)

    # 五模型对比
    results = train_all_models(X, y)

    # Stacking
    stacking = train_stacking(X, y)

    # 合并结果
    stack_row = pd.DataFrame([{'model': 'STACKING', 'R2_mean': stacking['R2_mean'],
                                'MAE_mean': stacking['MAE_mean'], 'RMSE_mean': stacking['RMSE_mean'],
                                'MAPE_mean': stacking['MAPE_mean'], 'train_time_s': 0}])
    plot_df = pd.concat([results, stack_row], ignore_index=True)

    # 雷达图
    os.makedirs(FIGURES_DIR, exist_ok=True)
    radar_path = os.path.join(FIGURES_DIR, 'model_radar_comparison.png')
    plot_radar_chart(plot_df, save_path=radar_path)
    safe_print(f"\n  [OK] 雷达图: {radar_path}")

    # 定量对比表
    safe_print(f"\n  {'Model':<18} {'R2':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'训练时间':>10}")
    safe_print(f"  {'─' * 75}")
    for _, row in plot_df.iterrows():
        safe_print(f"  {row['model']:<18} {row['R2_mean']:>8.4f} {row['MAE_mean']:>12,.0f} "
                   f"{row['RMSE_mean']:>12,.0f} {row['MAPE_mean']:>9.1f}% "
                   f"{row['train_time_s']:>8.1f}s")

    # 业务解读
    best_mae = plot_df['MAE_mean'].min()
    safe_print(f"\n[业务解读]")
    safe_print(f"  最优MAE = {best_mae:,.0f} 元: 模型对一辆车的估价平均偏差约{best_mae:,.0f}元")
    safe_print(f"  在二手车均价~30,000元的市场中, 误差率约{best_mae/30000*100:.1f}%")
    safe_print(f"  实际意义: 估价偏差在可接受范围, 可用作定价参考辅以人工复核")
    safe_print(f"\n  [OK] 4.3 模型对比完成")
    return plot_df


if __name__ == "__main__":
    run_model_comparison()
