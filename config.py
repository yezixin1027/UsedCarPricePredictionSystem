# config.py — 全局配置（项目根目录）
# =====================================

#
#
#   数据探索性分析 → 使用 TRAIN_PATH, TEST_PATH, FIGURES_DIR
#   数据预处理     → 使用 PROCESSED_DATA_DIR, MODEL_HYPERPARAMS
#   模型构建与评估 → 使用 MODEL_DIR, AVAILABLE_MODELS, RANDOM_SEED
#   结果分析       → 使用 FIGURES_DIR
#   系统实现       → 使用 MODEL_DIR
import os
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")

PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_features.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_features.csv")

# ============================================================
# 全局参数
# ============================================================
RANDOM_SEED = 42
CURRENT_YEAR = datetime.now().year  # 动态获取当前年份计算车龄
DEFAULT_CV_FOLDS = 5

# 目标变量
TARGET = "price"

# 可用模型列表 (5种范式 + Stacking集成)
# 线性:   ridge(L2), elastic_net(L1+L2)
# Bagging: random_forest
# Boosting: lightgbm
# 距离:   knn
# 集成:   stacking
AVAILABLE_MODELS = ["ridge", "elastic_net", "random_forest", "lightgbm", "knn"]

# 当前激活的模型（用于高级审计等）
ACTIVE_MODEL = "lightgbm"

# ============================================================
# 7 个模型的默认超参数字典（后续会被 Optuna 调优结果覆盖）
# ============================================================
MODEL_HYPERPARAMS = {
    # ---- 线性模型 ----
    "ridge": {
        "alpha": 1.0
    },
    "elastic_net": {
        "alpha": 0.1,
        "l1_ratio": 0.5,         # 0.5 = L1和L2各占一半
        "max_iter": 2000,
        "random_state": 42
    },
    # ---- 树模型 · Bagging ----
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "random_state": 42,
        "n_jobs": -1
    },
    # ---- 树模型 · Boosting ----
    "lightgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    },
    # ---- 距离模型 ----
    "knn": {
        "n_neighbors": 20,
        "weights": "distance",    # 近邻加权: 更近的点权重更大
        "p": 2,                   # 欧氏距离
        "n_jobs": -1
    }
}
