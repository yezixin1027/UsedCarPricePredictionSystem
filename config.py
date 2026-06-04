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
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
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
    "ridge": {
        "alpha": 5.0
    },
    "elastic_net": {
        "alpha": 0.01,
        "l1_ratio": 0.1,
        "max_iter": 5000,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "lightgbm": {
        "num_leaves": 94,
        "learning_rate": 0.035240102913883735,
        "min_child_samples": 14,
        "colsample_bytree": 0.5306905097710306,
        "subsample": 0.7623074715989433,
        "reg_alpha": 6.572406184977355,
        "reg_lambda": 2.5285087745914234,
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    },
    "knn": {
        "n_neighbors": 50,
        "weights": "uniform",
        "p": 2,
        "n_jobs": -1
    }
}
