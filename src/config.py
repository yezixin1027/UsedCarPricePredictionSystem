import os
from datetime import datetime

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")

# 全局参数
RANDOM_SEED = 42
CURRENT_YEAR = datetime.now().year  # 动态获取当前年份计算车龄

# 目标变量
TARGET = "price"