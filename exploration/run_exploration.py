# exploration/run_exploration.py
# =============================
# 论文第2章 · 数据探索性分析 — 阶段入口
#
# 按论文顺序依次执行:
#   s1: 数据集概览 (2.1)
#   s2: 目标变量分析 (2.2)
#   s3: 特征相关性热力图 (2.3)
#   s4: 特征变换策略对比 (2.4)
#   s5: 衍生特征有效性验证 (2.5)
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_exploration(skip_heavy=False):
    """执行第2章全部探索性数据分析"""
    start_time = time.time()
    print("=" * 70)
    print("  第2章: 数据探索性分析")
    print("=" * 70)

    # s1: 数据集概览 (2.1)
    print("\n" + "─" * 50)
    from exploration.s1_dataset_overview import run_dataset_overview
    run_dataset_overview()

    # s2: 目标变量分析 (2.2)
    print("\n" + "─" * 50)
    from exploration.s2_target_analysis import run_target_analysis
    run_target_analysis()

    # s3: 特征相关性 (2.3)
    print("\n" + "─" * 50)
    from exploration.s3_correlation_heatmap import run_correlation_analysis
    run_correlation_analysis()

    if not skip_heavy:
        # s4: 特征变换对比 (2.4)
        print("\n" + "─" * 50)
        from exploration.s4_feature_transform import run_feature_transform_analysis
        run_feature_transform_analysis()

        # s5: 衍生特征验证 (2.5)
        print("\n" + "─" * 50)
        from exploration.s5_feature_validation import run_feature_validation
        run_feature_validation()
    else:
        print("\n  [SKIP] s4, s5 (轻量模式)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  第2章完成 | 耗时: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--light', action='store_true', help='轻量模式, 跳过s4/s5')
    args = p.parse_args()
    run_exploration(skip_heavy=args.light)
