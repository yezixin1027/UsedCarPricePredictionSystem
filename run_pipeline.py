# run_pipeline.py
import os
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, PROCESSED_DATA_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer

if __name__ == "__main__":
    print("====== 阶段 1: 加载原始交易数据 ======")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 分离特征与标签
    X_train_raw = train_df.drop(columns=['price'])
    y_train_raw = train_df['price']
    X_test_raw = test_df.copy()

    print("====== 阶段 2: 启动防泄露高容错预处理 Pipeline ======")
    preprocessor = AdvancedUsedCarPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_test_clean = preprocessor.transform(X_test_raw)

    print("====== 阶段 3: 启动嵌套交叉验证特征工程 Pipeline ======")
    engineer = HighScoreFeatureEngineer()
    # 传入 y_train_raw 用于 OOF 目标编码
    X_train_final = engineer.fit_transform(X_train_clean, y_train_raw)
    X_test_final = engineer.transform(X_test_clean)

    # 拼回对数价格，形成最终的建模数据集
    train_final_output = X_train_final.copy()
    train_final_output['log_price'] = np.log1p(y_train_raw)

    print("====== 阶段 4: 持久化安全特征矩阵 ======")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_final_output.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_features.csv"), index=False)
    X_test_final.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_features.csv"), index=False)

    print(f"最终训练特征矩阵维度: {X_train_final.shape}")