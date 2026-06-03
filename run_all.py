#!/usr/bin/env python
# run_all.py — 二手车价格预测系统 · 全流程一键执行
# ==================================================
#
# 按论文章节顺序自动运行全部分析流程:
#
#   exploration/    数据探索性分析
#   preprocessing/  数据预处理与特征工程
#   modeling/       模型构建与评估
#   analysis/       结果分析与决策优化
#   application/    系统设计与实现 (需手动启动)
#
# Usage:
#   python run_all.py                      # 全部流程
#   python run_all.py --start 2            # 从数据预处理与特征工程开始
#   python run_all.py --start 3            # 从模型构建与评估开始 (跳过EDA)
#   python run_all.py --skip-tune          # 跳过超参数调优
#   python run_all.py --light              # 轻量模式 (跳过耗时步骤)
import os, sys, time, argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def main():
    parser = argparse.ArgumentParser(description='二手车价格预测系统 · 全流程一键执行')
    parser.add_argument('--start', type=int, default=2, choices=[2, 3, 4, 5],
                        help='起始章节 (default: 2)')
    parser.add_argument('--skip-tune', action='store_true', default=True,
                        help='跳过超参数调优 (default: True)')
    parser.add_argument('--tune', action='store_true',
                        help='运行超参数调优 (耗时约30-60分钟)')
    parser.add_argument('--light', action='store_true',
                        help='轻量模式 (跳过部分耗时验证)')
    parser.add_argument('--no-stacking', action='store_true',
                        help='跳过 Stacking 集成')
    args = parser.parse_args()

    total_start = time.time()
    safe_print("=" * 70)
    safe_print("  二手车价格预测系统 · 全流程执行")
    safe_print("  Used Car Price Prediction System — Full Pipeline")
    safe_print("=" * 70)
    safe_print(f"    数据探索性分析      → exploration/")
    safe_print(f"    数据预处理与特征工程 → preprocessing/")
    safe_print(f"    模型构建与评估      → modeling/")
    safe_print(f"    结果分析与决策优化   → analysis/")
    safe_print(f"    系统设计与实现      → application/")
    safe_print("=" * 70)

    # ================================================================
    # 数据探索性分析
    # ================================================================
    if args.start <= 2:
        safe_print(f"\n{'#' * 70}")
        safe_print(f"# 数据探索性分析")
        safe_print(f"{'#' * 70}")
        from exploration.run_exploration import run_exploration
        run_exploration(skip_heavy=args.light)

    # ================================================================
    # 数据预处理与特征工程
    # ================================================================
    if args.start <= 3:
        safe_print(f"\n{'#' * 70}")
        safe_print(f"# 数据预处理与特征工程")
        safe_print(f"{'#' * 70}")
        from preprocessing.run_preprocessing import run_preprocessing
        run_preprocessing()

        # 编码对比实验 (独立运行, 不阻塞主流程)
        try:
            from preprocessing.encoding_comparison import run_encoding_comparison
            run_encoding_comparison()
        except Exception as e:
            safe_print(f"  [WARNING] 编码对比实验失败: {e}")

    # ================================================================
    # 模型构建与评估
    # ================================================================
    if args.start <= 4:
        safe_print(f"\n{'#' * 70}")
        safe_print(f"# 模型构建与评估")
        safe_print(f"{'#' * 70}")
        from modeling.run_modeling import run_modeling
        run_modeling(skip_tune=not args.tune, skip_stacking=args.no_stacking)

    # ================================================================
    # 结果分析与决策优化
    # ================================================================
    if args.start <= 5:
        safe_print(f"\n{'#' * 70}")
        safe_print(f"#结果分析与决策优化")
        safe_print(f"{'#' * 70}")
        from analysis.run_analysis import run_analysis
        run_analysis()

    # ================================================================
    # 完成报告
    # ================================================================
    elapsed = time.time() - total_start
    safe_print(f"\n{'=' * 70}")
    safe_print(f"  全部流程执行完成!")
    safe_print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    safe_print(f"")
    safe_print(f"  输出文件:")
    safe_print(f"    图表:     reports/figures/ (20+ PNG)")
    safe_print(f"    模型:     models/ (5个模型权重)")
    safe_print(f"    特征矩阵: data/processed/ (train_features.csv)")
    safe_print(f"")
    safe_print(f"  下一步:")
    safe_print(f"    启动 Web 界面: streamlit run application/app.py")
    safe_print(f"    或: python application/run_app.py")
    safe_print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
