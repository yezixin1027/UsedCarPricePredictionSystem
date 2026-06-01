# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class HighScoreFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.brand_target_map_ = {}
        self.global_mean_ = None
        self.numerical_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']

    def fit(self, X, y):
        X_copy = X.copy()

        # 显式转换并强制对齐标签索引，防止多重交叉计算错位
        y_log = np.log1p(y).reset_index(drop=True)
        X_copy = X_copy.reset_index(drop=True)

        # 建立全局映射字典，供测试集静态转换无缝映射
        df_temp = X_copy.copy()
        df_temp['target'] = y_log
        self.brand_target_map_ = df_temp.groupby('brand')['target'].mean().to_dict()
        self.global_mean_ = y_log.mean()

        # 预先拟合包含“创新衍生复合特征”的标准化转换器
        extended_cols = self.numerical_cols + ['annual_milage']
        X_copy['annual_milage'] = X_copy['milage'] / (X_copy['car_age'] + 1)
        self.scaler.fit(X_copy[extended_cols])
        return self

    def transform(self, X, y=None):
        X_out = X.copy().reset_index(drop=True)

        # 1. 核心衍生创新：车龄-里程复合特征（年均行驶里程）
        X_out['annual_milage'] = X_out['milage'] / (X_out['car_age'] + 1)

        # 2. 亮点：五折嵌套隔离交叉验证目标编码（Out-of-fold Target Encoding），切断泄露通道
        if y is not None:
            y_log = np.log1p(y).reset_index(drop=True)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            oof_encoded = pd.Series(index=X_out.index, dtype=float)

            for train_idx, val_idx in kf.split(X_out):
                fold_train_x = X_out.iloc[train_idx]
                fold_train_y = y_log.iloc[train_idx]

                fold_df = fold_train_x.copy()
                fold_df['target'] = fold_train_y
                map_dict = fold_df.groupby('brand')['target'].mean().to_dict()

                oof_encoded.iloc[val_idx] = X_out.iloc[val_idx]['brand'].map(map_dict)
            X_out['brand_encoded'] = oof_encoded.fillna(self.global_mean_)
        else:
            X_out['brand_encoded'] = X_out['brand'].map(self.brand_target_map_).fillna(self.global_mean_)

        # 低基数名义特征独热编码 (One-Hot)
        ohe_cols = [c for c in ['fuel_type', 'accident_status'] if c in X_out.columns]
        if ohe_cols:
            X_out = pd.get_dummies(X_out, columns=ohe_cols, drop_first=True)

        # 3. 连续特征 Z-score 标准化转换
        extended_cols = self.numerical_cols + ['annual_milage']
        X_out[extended_cols] = self.scaler.transform(X_out[extended_cols])

        if 'brand' in X_out.columns:
            X_out = X_out.drop(columns=['brand'])

        # 强行确保输出的二值型矩阵转换成清晰数值
        for col in X_out.columns:
            if X_out[col].dtype == bool:
                X_out[col] = X_out[col].astype(int)

        return X_out