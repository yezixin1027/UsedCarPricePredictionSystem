# analysis/run_analysis.py
# ========================
# 论文第5章 · 结果分析与决策优化 — 阶段入口
#
#   s1: 预测结果分析 (5.1) — 预测vs实际散点图、误差分布、价格区间分析
#   s2: 业务建议 (5.2) — 基于模型洞察的二手车定价策略
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_analysis():
    print("=" * 70)
    print("  第5章: 结果分析与决策优化")
    print("=" * 70)

    from analysis.s1_prediction_analysis import run_prediction_analysis
    run_prediction_analysis()

    from analysis.s2_business_insights import run_business_insights
    run_business_insights()

    print(f"\n  [OK] 第5章完成")


if __name__ == "__main__":
    run_analysis()
