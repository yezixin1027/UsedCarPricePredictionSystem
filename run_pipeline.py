# run_pipeline.py
import os
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, PROCESSED_DATA_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer

if __name__ == "__main__":
    print("====== [阶段 1]: 开始加载二手车原始交易数据 ======")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 严格拆分原始自变量矩阵与响应变量
    X_train_raw = train_df.drop(columns=['price'])
    y_train_raw = train_df['price']
    X_test_raw = test_df.copy()

    print("====== [阶段 2]: 运行 AdvancedPreprocessor 预处理 Pipeline ======")
    preprocessor = AdvancedUsedCarPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_test_clean = preprocessor.transform(X_test_raw)

    print("====== [阶段 3]: 运行 HighScoreFeatureEngineer 特征工程 Pipeline ======")
    engineer = HighScoreFeatureEngineer()

    # 传入 y_train_raw 触发五折嵌套隔离目标编码机制
    X_train_final = engineer.fit_transform(X_train_clean, y_train_raw)
    X_test_final = engineer.transform(X_test_clean)

    # 拼回对数价格，形成完全干净、对齐、无泄露风险的持久化最终建模特征矩阵
    train_final_output = X_train_final.copy()
    train_final_output['log_price'] = np.log1p(y_train_raw).reset_index(drop=True)

    print("====== [阶段 4]: 持久化干净的工业级特征矩阵至硬盘 ======")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    train_final_output.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_features.csv"), index=False)
    X_test_final.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_features.csv"), index=False)

    print("\n" + "=" * 50)
    print(f"最终训练集特征矩阵维度: {train_final_output.shape}")
    print(f"最终测试集特征矩阵维度: {X_test_final.shape}")
    print("=" * 50)