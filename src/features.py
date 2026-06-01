# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class HighScoreFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=5, smoothing_weight=10):
        """
        n_splits: 交叉验证折数
        smoothing_weight: 目标编码平滑权重（拉普拉斯平滑思想），防止小样本品牌因极端价位产生过拟合噪音
        """
        self.n_splits = n_splits
        self.smoothing_weight = smoothing_weight
        self.scaler = StandardScaler()
        self.brand_target_map_ = {}
        self.global_mean_ = None
        self.numerical_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']

    def fit(self, X, y):
        X_copy = X.copy().reset_index(drop=True)
        y_log = np.log1p(y).reset_index(drop=True)

        self.global_mean_ = y_log.mean()

        # 计算训练集全局的品牌目标编码映射（融入平滑权重，防止孤本品牌泄露）
        df_temp = X_copy.copy()
        df_temp['target'] = y_log
        brand_stats = df_temp.groupby('brand')['target'].agg(['count', 'mean'])

        # 核心数学亮点：拉普拉斯平滑目标编码 (Smoothed Target Encoding)
        # 编码值 = (品牌样本量 * 品牌均值 + 平滑权重 * 全局均值) / (品牌样本量 + 平滑权重)
        smoothed_vals = (brand_stats['count'] * brand_stats['mean'] + self.smoothing_weight * self.global_mean_) / (
                    brand_stats['count'] + self.smoothing_weight)
        self.brand_target_map_ = smoothed_vals.to_dict()

        # 预先拟合包含衍生复合特征的标准化转换器
        extended_cols = self.numerical_cols + ['annual_milage']
        X_copy['annual_milage'] = X_copy['milage'] / (X_copy['car_age'] + 1)
        self.scaler.fit(X_copy[extended_cols])
        return self

    def transform(self, X, y=None):
        X_out = X.copy().reset_index(drop=True)

        # 1. 核心衍生创新特征：年均行驶里程数（物理磨损强度）
        X_out['annual_milage'] = X_out['milage'] / (X_out['car_age'] + 1)

        # 2. 五折嵌套隔离交叉验证目标编码机制
        if y is not None:
            y_log = np.log1p(y).reset_index(drop=True)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            oof_encoded = pd.Series(index=X_out.index, dtype=float)

            for train_idx, val_idx in kf.split(X_out):
                fold_train_x = X_out.iloc[train_idx]
                fold_train_y = y_log.iloc[train_idx]

                fold_df = fold_train_x.copy()
                fold_df['target'] = fold_train_y

                # 局部折内同样引入平滑权重计算
                f_stats = fold_df.groupby('brand')['target'].agg(['count', 'mean'])
                f_smoothed = (f_stats['count'] * f_stats['mean'] + self.smoothing_weight * self.global_mean_) / (
                            f_stats['count'] + self.smoothing_weight)
                f_map = f_smoothed.to_dict()

                oof_encoded.iloc[val_idx] = X_out.iloc[val_idx]['brand'].map(f_map)

            X_out['brand_encoded'] = oof_encoded.fillna(self.global_mean_)
        else:
            # 测试集直接无缝静态映射
            X_out['brand_encoded'] = X_out['brand'].map(self.brand_target_map_).fillna(self.global_mean_)

        # 低基数类别特征独热编码化
        ohe_cols = [c for c in ['fuel_type', 'accident_status'] if c in X_out.columns]
        if ohe_cols:
            # 强行限定 dtype=int，从源头上斩断输出 True/False 布尔类型的隐患
            X_out = pd.get_dummies(X_out, columns=ohe_cols, drop_first=True, dtype=int)

        # 3. 连续特征 Z-score 标准化转换
        extended_cols = self.numerical_cols + ['annual_milage']
        X_out[extended_cols] = self.scaler.transform(X_out[extended_cols])

        if 'brand' in X_out.columns:
            X_out = X_out.drop(columns=['brand'])

        # 格式安全终审：确保全矩阵数据类型无缝契合各流派算法后端
        for col in X_out.columns:
            if X_out[col].dtype == bool:
                X_out[col] = X_out[col].astype(int)

        return X_out