# modeling/s4_model_explainability.py
# ===================================
# 论文 4.4 最佳模型深度分析
#
# 1. SHAP 可解释性分析 (全局+局部):
#    - Summary Plot: 全局特征重要性排序
#    - Importance Bar: 特征贡献量化
#    - Dependence Plot: 关键特征边际效应
#    - Waterfall Plot: 单样本预测分解
#
# 2. 高级审计:
#    - Top-10 极端误差样本分析
#    - 品牌/价格区间公平性评估
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_TRAIN_PATH, MODEL_DIR


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_model_explainability():
    """运行 4.4 最佳模型深度分析"""
    safe_print("=" * 60)
    safe_print("  4.4 最佳模型深度分析")
    safe_print("=" * 60)

    # ---- Part A: SHAP 可解释性 ----
    safe_print("\n[Part A] SHAP 可解释性分析")
    from src.explainability import run_shap_analysis
    run_shap_analysis()

    # ---- Part B: 高级审计 ----
    safe_print("\n[Part B] 高级审计 (Top-10误差 + 公平性)")
    from src.evaluate import run_advanced_audit
    run_advanced_audit()

    safe_print(f"\n  [OK] 4.4 深度分析完成")


if __name__ == "__main__":
    run_model_explainability()
