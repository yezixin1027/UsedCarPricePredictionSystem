# application/backend/pipeline_loader.py
# 加载训练阶段保存的 Pipeline 对象（单例模式）

import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_DIR, PROCESSED_TRAIN_PATH

# 模块级缓存
_preprocessor = None
_feature_engineer = None
_feature_selector = None
_feature_names = None
_brand_list = None
_model_list = None
_model = None
_shap_explainer = None
_stats_cache = None


def _resolve_path(filename):
    return os.path.join(MODEL_DIR, filename)


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        path = _resolve_path('preprocessor.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"预处理器不存在: {path}，请先运行 preprocessing/run_preprocessing.py")
        with open(path, 'rb') as f:
            _preprocessor = pickle.load(f)
    return _preprocessor


def get_feature_engineer():
    global _feature_engineer
    if _feature_engineer is None:
        path = _resolve_path('feature_engineer.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征工程器不存在: {path}")
        with open(path, 'rb') as f:
            _feature_engineer = pickle.load(f)
    return _feature_engineer


def get_feature_selector():
    global _feature_selector
    if _feature_selector is None:
        path = _resolve_path('feature_selector.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征选择器不存在: {path}")
        with open(path, 'rb') as f:
            _feature_selector = pickle.load(f)
    return _feature_selector


def get_feature_names():
    global _feature_names
    if _feature_names is None:
        path = _resolve_path('feature_names.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征列名不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            _feature_names = json.load(f)
    return _feature_names


def get_brand_list():
    global _brand_list
    if _brand_list is None:
        path = _resolve_path('brand_list.json')
        with open(path, 'r', encoding='utf-8') as f:
            _brand_list = json.load(f)
    return _brand_list


def get_model_list():
    global _model_list
    if _model_list is None:
        path = _resolve_path('model_list.json')
        with open(path, 'r', encoding='utf-8') as f:
            _model_list = json.load(f)
    return _model_list


def get_model():
    global _model
    if _model is None:
        path = _resolve_path('lightgbm_model.pkl')
        if not os.path.exists(path):
            path = _resolve_path('xgboost_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, 'rb') as f:
            _model = pickle.load(f)
    return _model


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        import shap
        import pandas as pd

        model = get_model()
        # 用处理后训练数据做背景
        train_path = PROCESSED_TRAIN_PATH
        df = pd.read_csv(train_path)
        feature_names = get_feature_names()
        X_bg = df[feature_names].sample(n=min(500, len(df)), random_state=42).values.astype(float)

        _shap_explainer = shap.TreeExplainer(model, X_bg)
    return _shap_explainer


def get_stats():
    global _stats_cache
    if _stats_cache is None:
        import pandas as pd

        train_path = PROCESSED_TRAIN_PATH
        df = pd.read_csv(train_path)
        prices = np.expm1(df['log_price'].values)

        _stats_cache = {
            'avg_price': float(np.mean(prices)),
            'total_samples': len(df),
            'model_r2': 0.6622,   # 来自 model_comparison.csv
            'model_mae': 17189.0,  # 来自 model_comparison.csv
        }
    return _stats_cache
