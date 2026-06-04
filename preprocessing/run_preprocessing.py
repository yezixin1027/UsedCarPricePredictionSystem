# preprocessing/run_preprocessing.py
# ==============================
# 数据预处理与特征工程 — 阶段入口
#
# 顺序依次执行:
#   s1: 数据清洗与异常值处理 (3.1)
#   s2: 分类特征编码方式对比 (3.2)
#   s3: 特征工程 — 衍生+交互+编码+标准化 (3.3+3.4)
#   s4: 特征选择 — VIF + MI两阶段筛选 (3.5)
#
# 执行完毕后生成 data/processed/train_features.csv 和 test_features.csv
import os, sys, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, TEST_PATH, PROCESSED_DATA_DIR
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer
from src.selection import FeatureSelector


def safe_print(msg):
    try: print(msg)
    except UnicodeEncodeError: print(msg.encode('gbk', errors='replace').decode('gbk'))


def run_preprocessing():
    """全部数据预处理与特征工程流程"""
    start_time = time.time()
    safe_print("=" * 70)
    safe_print("  第3章: 数据预处理与特征工程")
    safe_print("=" * 70)

    # ---- s1: 数据加载与环境校验 (3.1 前置) ----
    safe_print("\n[3.1] 数据加载与环境校验")
    for path, label in [(TRAIN_PATH, "训练集"), (TEST_PATH, "测试集")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{label}] 文件不存在: {path}")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train_raw = train_df.drop(columns=['price'])
    y_train_raw = train_df['price']
    X_test_raw = test_df.copy()
    safe_print(f"  训练集: {train_df.shape} | 测试集: {test_df.shape}")

    # ---- s1: 数据清洗 (3.1) ----
    safe_print("\n[3.1] 数据清洗与异常值处理 (AdvancedUsedCarPreprocessor)")
    preprocessor = AdvancedUsedCarPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_test_clean = preprocessor.transform(X_test_raw)
    safe_print(f"  清洗后: {X_train_clean.shape}")

    # ---- s3: 特征工程 (3.3) ----
    safe_print("\n[3.3] 特征工程 (HighScoreFeatureEngineer)")
    safe_print("  衍生特征(4): annual_milage, power_density, car_age_squared, hp_per_year")
    safe_print("  交互特征: 无 (brand_x_milage 经 MI 验证无效, 已移除)")
    safe_print("  目标编码(2): brand_encoded, model_encoded")
    engineer = HighScoreFeatureEngineer()
    X_train_feat = engineer.fit_transform(X_train_clean, y_train_raw)
    X_test_feat = engineer.transform(X_test_clean)
    safe_print(f"  特征工程后: {X_train_feat.shape[1]} 维")

    # ---- s4: 特征选择 (3.4) ----
    safe_print("\n[3.4] 特征选择 (VIF共线性 + MI互信息)")
    selector = FeatureSelector(vif_threshold=10.0, mi_percentile_factor=0.05, verbose=True)
    try:
        X_train_sel = selector.fit_transform(X_train_feat, y_train_raw)
        X_test_sel = selector.transform(X_test_feat)
    except Exception:
        safe_print("[WARNING] 特征选择失败, 回退到全特征")
        X_train_sel, X_test_sel = X_train_feat, X_test_feat

    safe_print(f"  筛选前: {X_train_feat.shape[1]} -> 筛选后: {X_train_sel.shape[1]} "
               f"(剔除 {X_train_feat.shape[1] - X_train_sel.shape[1]} 个)")

    # ---- 持久化：特征矩阵 ----
    safe_print("\n[持久化] 保存特征矩阵到 data/processed/")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_out = X_train_sel.copy()
    train_out['log_price'] = np.log1p(y_train_raw).reset_index(drop=True)
    train_out.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_features.csv"), index=False)
    X_test_sel.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_features.csv"), index=False)

    # ---- 持久化：Pipeline 对象（供 FastAPI 推理使用） ----
    import json, pickle
    from config import MODEL_DIR

    safe_print("\n[持久化] 保存 Pipeline 对象到 models/")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 预处理器
    with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    safe_print("  [OK] preprocessor.pkl")

    # 2. 特征工程器（含 brand_target_map_ / model_target_map_ / scaler）
    with open(os.path.join(MODEL_DIR, 'feature_engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
    safe_print("  [OK] feature_engineer.pkl")

    # 3. 特征选择器（含保留特征名列表）
    with open(os.path.join(MODEL_DIR, 'feature_selector.pkl'), 'wb') as f:
        pickle.dump(selector, f)
    safe_print("  [OK] feature_selector.pkl")

    # 4. 最终特征列名（JSON，便于调试和前端读取）
    feature_names = X_train_sel.columns.tolist()
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    safe_print(f"  [OK] feature_names.json ({len(feature_names)} 个特征)")

    # 5. 品牌列表 + 车型映射（供前端下拉框）
    brand_list = sorted(train_df['brand'].dropna().unique().tolist())
    with open(os.path.join(MODEL_DIR, 'brand_list.json'), 'w', encoding='utf-8') as f:
        json.dump(brand_list, f, ensure_ascii=False, indent=2)
    safe_print(f"  [OK] brand_list.json ({len(brand_list)} 个品牌)")

    # 车型按品牌分组（只保留样本量>=10的车型）
    model_grouped = {}
    for b in brand_list:
        brand_models = train_df[train_df['brand'] == b]['model'].value_counts()
        valid_models = brand_models[brand_models >= 10].index.tolist()
        if valid_models:
            model_grouped[b] = sorted(valid_models)
    with open(os.path.join(MODEL_DIR, 'model_list.json'), 'w', encoding='utf-8') as f:
        json.dump(model_grouped, f, ensure_ascii=False, indent=2)
    safe_print(f"  [OK] model_list.json ({sum(len(v) for v in model_grouped.values())} 个车型)")

    elapsed = time.time() - start_time
    safe_print(f"\n{'=' * 60}")
    safe_print(f"  第3章完成 | 耗时: {elapsed:.1f}s")
    safe_print(f"  最终特征维度: {X_train_sel.shape[1]}")
    safe_print(f"{'=' * 60}")
    return X_train_sel, X_test_sel


if __name__ == "__main__":
    run_preprocessing()
