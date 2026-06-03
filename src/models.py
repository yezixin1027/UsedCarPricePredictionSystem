# src/models.py
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from config import MODEL_HYPERPARAMS


class UsedCarModelFactory:
    """可配置式二手车预测模型工厂 (Factory Pattern)

    6 种模型 (5 种范式 + 集成):
      线性·L2:     ridge
      线性·L1+L2:  elastic_net
      Bagging树:   random_forest
      Boosting树:  lightgbm
      距离:        knn
      集成:        stacking (Ridge+EN+RF+LGB+KNN → ElasticNet)
    """

    @staticmethod
    def create_model(model_name: str, **custom_kwargs):
        model_name = model_name.lower().strip()
        params = MODEL_HYPERPARAMS.get(model_name, {}).copy()
        params.update(custom_kwargs)

        if model_name == "ridge":
            return Ridge(**params)
        elif model_name == "elastic_net":
            return ElasticNet(**params)
        elif model_name == "random_forest":
            return RandomForestRegressor(**params)
        elif model_name == "lightgbm":
            return lgb.LGBMRegressor(**params)
        elif model_name == "knn":
            return KNeighborsRegressor(**params)
        elif model_name == "stacking":
            # 5种范式基模型: 线性(L2+L1L2) + Bagging + Boosting + 距离
            ridge_p  = MODEL_HYPERPARAMS.get('ridge', {}).copy()
            en_p     = MODEL_HYPERPARAMS.get('elastic_net', {}).copy()
            rf_p     = MODEL_HYPERPARAMS.get('random_forest', {}).copy()
            lgb_p    = MODEL_HYPERPARAMS.get('lightgbm', {}).copy()
            knn_p    = MODEL_HYPERPARAMS.get('knn', {}).copy()

            lgb_p.pop('verbose', None)
            lgb_p.pop('n_jobs', None)
            rf_p.pop('n_jobs', None)
            knn_p.pop('n_jobs', None)

            base_estimators = [
                ('ridge', Ridge(**ridge_p)),
                ('en',    ElasticNet(**en_p)),
                ('rf',    RandomForestRegressor(**rf_p)),
                ('lgb',   lgb.LGBMRegressor(**lgb_p)),
                ('knn',   KNeighborsRegressor(**knn_p)),
            ]

            return StackingRegressor(
                estimators=base_estimators,
                final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5,
                                            max_iter=5000, random_state=42),
                cv=5, n_jobs=-1, passthrough=True
            )
        else:
            raise ValueError(f"[Error] 未注册模型: {model_name}")
