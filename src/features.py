# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class HighScoreFeatureEngineer(BaseEstimator, TransformerMixin):
    """高评分特征工程器（精简版，仅保留通过 VIF+MI 验证的特征）。

    1. 衍生特征 (4个): annual_milage, power_density, car_age_squared, hp_per_year
    2. 目标编码: brand + model (五折嵌套OOF, 拉普拉斯平滑)
    3. One-Hot编码: fuel_type, accident_status (低基数分类变量)
    4. Z-score标准化: 连续数值特征无量纲化
    """

    def __init__(self, n_splits=5, smoothing_weight=10):
        self.n_splits = n_splits
        self.smoothing_weight = smoothing_weight
        self.scaler = StandardScaler()
        self.brand_target_map_ = {}
        self.model_target_map_ = {}
        self.global_mean_ = None
        self.numerical_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']
        # 扩展衍生特征列表
        self.derived_cols = ['annual_milage', 'power_density',
                             'car_age_squared', 'hp_per_year']

    def _build_derived_features(self, X):
        """构建衍生复合特征（共 4 个，均通过 VIF+MI 筛选验证有效）。

        - annual_milage: 年均行驶里程（磨损强度）
        - power_density: 马力密度/升功率（性能溢价指标）
        - car_age_squared: 车龄平方（捕获非线性折旧加速效应）
        - hp_per_year: 年均马力保有量（发动机老化速度，XGBoost 最重要特征）
        """
        X_out = X.copy()

        # === 基础衍生特征 ===
        # ① 年均行驶里程（物理磨损强度）
        X_out['annual_milage'] = X_out['milage'] / (X_out['car_age'] + 1)

        # ② 马力密度（比功率 / 升功率）
        X_out['power_density'] = X_out['engine_hp'] / X_out['engine_liter'].replace(0, np.nan)
        pd_fallback = X_out['engine_hp'].median() / max(X_out['engine_liter'].median(), 0.1)
        X_out['power_density'] = X_out['power_density'].replace(
            [np.inf, -np.inf], np.nan).fillna(pd_fallback)

        # === 非线性特征 ===
        # ③ 车龄平方（捕获折旧加速 / 减缓效应，前 3 年折旧最快）
        X_out['car_age_squared'] = X_out['car_age'] ** 2

        # ④ 年均马力保有量（衡量发动机老化速度）
        X_out['hp_per_year'] = X_out['engine_hp'] / (X_out['car_age'] + 1)

        return X_out

    def _add_interaction_features(self, X):
        """交互特征构建（当前版本无已验证有效的交互项，保留方法留作扩展）。

        之前尝试的 brand_x_milage / brand_x_car_age / brand_x_power
        均被 VIF+MI 筛选淘汰，如需新增交互特征请在此添加。
        """
        return X

    def _smooth_encode(self, group_stats, global_mean, weight):
        """拉普拉斯平滑目标编码的通用计算。

        编码值 = (n * mean + weight * global_mean) / (n + weight)
        """
        return (group_stats['count'] * group_stats['mean'] + weight * global_mean) / \
               (group_stats['count'] + weight)

    def fit(self, X, y):
        X_copy = X.copy().reset_index(drop=True)
        y_log = np.log1p(y)

        self.global_mean_ = y_log.mean()

        # ---- 品牌目标编码映射（融入平滑权重）----
        df_temp = X_copy.copy()
        df_temp['target'] = y_log
        brand_stats = df_temp.groupby('brand')['target'].agg(['count', 'mean'])
        self.brand_target_map_ = self._smooth_encode(
            brand_stats, self.global_mean_, self.smoothing_weight).to_dict()

        # ---- 车型目标编码映射（新增）----
        if 'model' in df_temp.columns:
            model_stats = df_temp.groupby('model')['target'].agg(['count', 'mean'])
            self.model_target_map_ = self._smooth_encode(
                model_stats, self.global_mean_, self.smoothing_weight).to_dict()

        # 预先拟合标准化转换器（包含全部衍生特征）
        X_copy = self._build_derived_features(X_copy)
        extended_cols = self.numerical_cols + self.derived_cols
        self.scaler.fit(X_copy[extended_cols])
        return self

    def transform(self, X, y=None):
        X_out = X.copy().reset_index(drop=True)

        # 1. 构建衍生复合特征
        X_out = self._build_derived_features(X_out)

        # 2. 五折嵌套隔离 OOF 目标编码 — brand
        if y is not None:
            y_log = np.log1p(y).reset_index(drop=True)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            # --- brand OOF 编码 ---
            oof_brand = pd.Series(index=X_out.index, dtype=float)
            for train_idx, val_idx in kf.split(X_out):
                fold_train_x = X_out.iloc[train_idx]
                fold_train_y = y_log.iloc[train_idx]
                fold_df = fold_train_x.copy()
                fold_df['target'] = fold_train_y
                f_stats = fold_df.groupby('brand')['target'].agg(['count', 'mean'])
                f_smoothed = self._smooth_encode(
                    f_stats, self.global_mean_, self.smoothing_weight)
                oof_brand.iloc[val_idx] = X_out.iloc[val_idx]['brand'].map(f_smoothed)
            X_out['brand_encoded'] = oof_brand.fillna(self.global_mean_)

            # --- model OOF 编码（新增）---
            if 'model' in X_out.columns:
                oof_model = pd.Series(index=X_out.index, dtype=float)
                # 为 model 列单独做一次 KFold（与 brand 共用同样的划分）
                kf2 = KFold(n_splits=self.n_splits, shuffle=True, random_state=43)
                for train_idx, val_idx in kf2.split(X_out):
                    fold_train_x = X_out.iloc[train_idx]
                    fold_train_y = y_log.iloc[train_idx]
                    fold_df = fold_train_x.copy()
                    fold_df['target'] = fold_train_y
                    f_stats = fold_df.groupby('model')['target'].agg(['count', 'mean'])
                    f_smoothed = self._smooth_encode(
                        f_stats, self.global_mean_, self.smoothing_weight * 2)  # model 样本更少，更大平滑
                    oof_model.iloc[val_idx] = X_out.iloc[val_idx]['model'].map(f_smoothed)
                X_out['model_encoded'] = oof_model.fillna(self.global_mean_)
        else:
            # 测试集：直接静态映射
            X_out['brand_encoded'] = X_out['brand'].map(
                self.brand_target_map_).fillna(self.global_mean_)
            if 'model' in X_out.columns:
                X_out['model_encoded'] = X_out['model'].map(
                    self.model_target_map_).fillna(self.global_mean_)

        # 3. 构建交互特征（必须在 brand_encoded 生成之后）
        X_out = self._add_interaction_features(X_out)

        # 4. 低基数类别特征独热编码化
        ohe_cols = [c for c in ['fuel_type', 'accident_status'] if c in X_out.columns]
        if ohe_cols:
            X_out = pd.get_dummies(X_out, columns=ohe_cols, drop_first=True, dtype=int)

        # 5. 连续特征 Z-score 标准化
        extended_cols = self.numerical_cols + self.derived_cols
        X_out[extended_cols] = self.scaler.transform(X_out[extended_cols])

        # 6. 清理文本列（已被编码替代）
        drop_text_cols = ['brand', 'model']
        X_out = X_out.drop(columns=[c for c in drop_text_cols if c in X_out.columns])

        # 7. 格式安全终审
        for col in X_out.columns:
            if X_out[col].dtype == bool:
                X_out[col] = X_out[col].astype(int)

        return X_out
