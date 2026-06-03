# preprocessing/data_cleaning.py
# ==================================
# 3.1 数据清洗与异常值处理
#
# 调用 src/preprocess.py 中的 AdvancedUsedCarPreprocessor:
#   - 品牌名称标准化 + 稀有品牌归拢
#   - 正则引擎特征抽取 (HP, 排量)
#   - 品牌级局部 IQR 截尾 (防豪车误杀)
#   - MAR 业务联动缺失值填充
#   - 车龄计算 + 文本列裁剪
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH
from src.preprocess import AdvancedUsedCarPreprocessor


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_data_cleaning():
    """运行 3.1 数据清洗演示"""
    safe_print("=" * 60)
    safe_print("  3.1 数据清洗与异常值处理")
    safe_print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    preprocessor = AdvancedUsedCarPreprocessor()

    safe_print("\n[处理前]")
    safe_print(f"  样本量: {len(train_df):,}")
    safe_print(f"  缺失值统计:")
    for col in train_df.columns:
        na = train_df[col].isna().sum()
        if na > 0:
            safe_print(f"    {col}: {na} ({na/len(train_df)*100:.1f}%)")

    X_clean = preprocessor.fit_transform(train_df.drop(columns=['price']))

    safe_print(f"\n[处理后]")
    safe_print(f"  维度: {X_clean.shape}")
    safe_print(f"  保留列: {list(X_clean.columns)}")
    safe_print(f"  各列缺失值: {X_clean.isna().sum().sum()} (应为0)")

    # 展示品牌处理结果
    if 'brand' in X_clean.columns:
        top_brands = X_clean['brand'].value_counts().head(10)
        safe_print(f"\n  Top-10品牌: {dict(top_brands)}")

    safe_print("\n[处理依据说明]")
    safe_print("  1. 品牌级IQR截尾: 不同品牌里程分布差异大(豪车vs代步车), 全局阈值会误杀")
    safe_print("  2. MAR填充: Tesla缺失fuel_type → 业务规则填充Electric")
    safe_print("  3. 稀有品牌归拢: <15样本的品牌合并为Other, 防止未知品牌预测崩溃")
    safe_print("  4. 引擎文本抽取: 正则提取HP和排量, 转为数值特征")

    safe_print(f"\n  [OK] 3.1 数据清洗完成")
    return X_clean


if __name__ == "__main__":
    run_data_cleaning()
