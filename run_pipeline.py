# run_pipeline.py
"""
二手车价格预测系统 · 主 Pipeline 执行脚本
==========================================

阶段划分:
  阶段 1  — 数据加载
  阶段 2  — 高级预处理 (AdvancedUsedCarPreprocessor)
  阶段 3  — 特征工程 (HighScoreFeatureEngineer)
  阶段 3.5 — 特征选择与降维 (FeatureSelector: VIF + 互信息)
  阶段 4  — 持久化输出

Usage:
    python run_pipeline.py
"""
import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, PROCESSED_DATA_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer
from src.selection import FeatureSelector


def log_stage(msg: str) -> None:
    """统一的分阶段日志输出，防止 Windows GBK 编码崩溃。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_pipeline() -> None:
    """按顺序执行全部 Pipeline 阶段，任一步骤异常均提前终止并输出诊断信息。"""
    start_time = time.time()

    # ====================================================================
    # 阶段 0: 环境校验
    # ====================================================================
    log_stage("====== [阶段 0]: 环境校验 ======")
    for path, label in [(TRAIN_PATH, "训练集"), (TEST_PATH, "测试集")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{label}] 文件不存在: {path}\n"
                                    f"  请将 train.csv 与 test.csv 放入 data/raw/ 目录")
    log_stage(f"  [OK] 训练集: {TRAIN_PATH}")
    log_stage(f"  [OK] 测试集: {TEST_PATH}")

    # ====================================================================
    # 阶段 1: 数据加载
    # ====================================================================
    log_stage("\n====== [阶段 1]: 开始加载二手车原始交易数据 ======")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except Exception as e:
        raise RuntimeError(f"CSV 文件读取失败: {e}") from e

    # 严格拆分原始自变量矩阵与响应变量
    X_train_raw = train_df.drop(columns=['price'])
    y_train_raw = train_df['price']
    X_test_raw = test_df.copy()

    log_stage(f"  训练集维度: {train_df.shape}")
    log_stage(f"  测试集维度: {test_df.shape}")

    # ====================================================================
    # 阶段 2: 高级预处理
    # ====================================================================
    log_stage("\n====== [阶段 2]: 运行 AdvancedPreprocessor 预处理 Pipeline ======")
    try:
        preprocessor = AdvancedUsedCarPreprocessor()
        X_train_clean = preprocessor.fit_transform(X_train_raw)
        X_test_clean = preprocessor.transform(X_test_raw)
    except Exception as e:
        log_stage(f"\n[ERROR] 预处理阶段失败:")
        traceback.print_exc()
        raise RuntimeError(f"预处理失败: {e}") from e

    log_stage(f"  清洗后训练集维度: {X_train_clean.shape}")
    log_stage(f"  清洗后测试集维度: {X_test_clean.shape}")

    # ====================================================================
    # 阶段 3: 特征工程
    # ====================================================================
    log_stage("\n====== [阶段 3]: 运行 HighScoreFeatureEngineer 特征工程 Pipeline ======")
    try:
        engineer = HighScoreFeatureEngineer()
        # 传入 y_train_raw 触发五折嵌套隔离目标编码机制
        X_train_final = engineer.fit_transform(X_train_clean, y_train_raw)
        X_test_final = engineer.transform(X_test_clean)
    except Exception as e:
        log_stage(f"\n[ERROR] 特征工程阶段失败:")
        traceback.print_exc()
        raise RuntimeError(f"特征工程失败: {e}") from e

    log_stage(f"  特征工程后训练集维度: {X_train_final.shape}")

    # ====================================================================
    # 阶段 3.5: 特征选择与降维
    # ====================================================================
    log_stage(f"\n====== [阶段 3.5]: 特征选择与降维 (VIF + 互信息) ======")
    try:
        selector = FeatureSelector(vif_threshold=10.0, mi_percentile_factor=0.05, verbose=True)
        X_train_selected = selector.fit_transform(X_train_final, y_train_raw)
        X_test_selected = selector.transform(X_test_final)
    except Exception as e:
        log_stage(f"\n[WARNING] 特征选择阶段失败，回退到全特征矩阵:")
        traceback.print_exc()
        X_train_selected = X_train_final
        X_test_selected = X_test_final
        log_stage(f"  已回退，继续执行后续阶段")

    log_stage(f"\n  特征选择前维度: {X_train_final.shape[1]}")
    log_stage(f"  特征选择后维度: {X_train_selected.shape[1]}")
    log_stage(f"  剔除特征数:     {X_train_final.shape[1] - X_train_selected.shape[1]}")

    # ---- 输出剔除报告 ----
    try:
        report_df = selector.get_report()
        if len(report_df) > 0:
            log_stage(f"\n  详细剔除报告:")
            for _, row in report_df.iterrows():
                log_stage(f"    - [{row['round']}] {row['feature']}: {row['reason']}")
        else:
            log_stage(f"\n  [OK] 所有特征均通过筛选, 无需剔除")
    except Exception:
        pass  # 报告生成非关键路径

    # ====================================================================
    # 阶段 4: 持久化输出
    # ====================================================================
    log_stage(f"\n====== [阶段 4]: 持久化干净的工业级特征矩阵至硬盘 ======")
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # 拼回对数价格，形成完全干净、对齐、无泄露风险的持久化最终建模特征矩阵
        train_final_output = X_train_selected.copy()
        train_final_output['log_price'] = np.log1p(y_train_raw).reset_index(drop=True)

        train_final_output.to_csv(
            os.path.join(PROCESSED_DATA_DIR, "train_features.csv"), index=False)
        X_test_selected.to_csv(
            os.path.join(PROCESSED_DATA_DIR, "test_features.csv"), index=False)
    except Exception as e:
        raise RuntimeError(f"持久化输出失败，请检查磁盘空间与目录权限: {e}") from e

    # ====================================================================
    # 阶段 5: 完成报告
    # ====================================================================
    elapsed = time.time() - start_time
    log_stage("\n" + "=" * 60)
    log_stage(f"  Pipeline 执行完成  |  耗时: {elapsed:.1f}s")
    log_stage(f"  最终训练集特征矩阵维度: {train_final_output.shape}")
    log_stage(f"  最终测试集特征矩阵维度: {X_test_selected.shape}")
    log_stage(f"  输出目录: {PROCESSED_DATA_DIR}")
    log_stage("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"[FATAL] Pipeline 因严重错误终止: {e}")
        print(f"{'=' * 60}")
        sys.exit(1)
