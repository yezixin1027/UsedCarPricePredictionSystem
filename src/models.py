# src/models.py
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from src.config import MODEL_HYPERPARAMS


class UsedCarModelFactory:
    """可配置式二手车预测模型工厂 (Factory Pattern)

    支持 5 种独立模型 + Stacking 集成：
    - ridge:        岭回归 (L2 正则化，基线模型)
    - random_forest: 随机森林 (Bagging 集成)
    - lightgbm:     LightGBM (直方图 Boosting)
    - xgboost:      XGBoost (预排序 Boosting)
    - catboost:     CatBoost (有序 Boosting)
    - stacking:     Stacking 集成 (LGB+XGB+Cat 基模型 + ElasticNet 元学习器)
    """

    @staticmethod
    def create_model(model_name: str, **custom_kwargs):
        """根据模型名称动态实例化算法对象，允许外部传入覆盖参数"""
        model_name = model_name.lower().strip()
        params = MODEL_HYPERPARAMS.get(model_name, {}).copy()
        params.update(custom_kwargs)

        if model_name == "ridge":
            return Ridge(**params)
        elif model_name == "random_forest":
            return RandomForestRegressor(**params)
        elif model_name == "lightgbm":
            return lgb.LGBMRegressor(**params)
        elif model_name == "xgboost":
            return xgb.XGBRegressor(**params)
        elif model_name == "catboost":
            return cb.CatBoostRegressor(**params)
        elif model_name == "stacking":
            # 使用当前最优参数构建基模型
            lgb_params = MODEL_HYPERPARAMS.get('lightgbm', {}).copy()
            xgb_params = MODEL_HYPERPARAMS.get('xgboost', {}).copy()
            cat_params = MODEL_HYPERPARAMS.get('catboost', {}).copy()

            # 清理子进程冲突参数
            for p in [lgb_params, xgb_params]:
                p.pop('verbose', None)
                p.pop('n_jobs', None)
            cat_params['verbose'] = False

            base_estimators = [
                ('lgb', lgb.LGBMRegressor(**lgb_params)),
                ('xgb', xgb.XGBRegressor(**xgb_params)),
                ('cat', cb.CatBoostRegressor(**cat_params)),
            ]

            # 元学习器: ElasticNet (L1+L2 正则化) — 兼顾特征选择与稳定性
            return StackingRegressor(
                estimators=base_estimators,
                final_estimator=ElasticNet(
                    alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
                cv=5,
                n_jobs=-1,
                passthrough=True  # 原始特征直达元学习器
            )
        else:
            raise ValueError(f"[Error] 未注册模型: {model_name}")
