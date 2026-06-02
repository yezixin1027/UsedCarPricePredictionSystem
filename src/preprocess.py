# src/preprocess.py
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import CURRENT_YEAR

class AdvancedUsedCarPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, rare_brand_threshold=15, iqr_multiplier=3.0):
        """
        rare_brand_threshold: 样本量少于该值的品牌收拢归类为 'Other'，防范未知品牌带来的预测崩溃
        iqr_multiplier: 局部截断设定为3.0（极值阈值），防止粗暴“误杀”高性能或限量版豪车
        """
        self.rare_brand_threshold = rare_brand_threshold
        self.iqr_multiplier = iqr_multiplier
        self.hp_median_ = None
        self.liter_median_ = None
        self.frequent_brands_ = None
        self.brand_iqr_bounds_ = {}  # 严格在训练集拟合时存储各品牌的局部异常值边界

    def _standardize_brands(self, brand_series):
        """清洗文本拼写歧义与非标准缩写，保留原始 NaN"""
        result = brand_series.astype(str).str.strip()
        result = result.replace({
            'Land': 'Land Rover',
            'Mercedes': 'Mercedes-Benz',
            'VW': 'Volkswagen',
            'Chevy': 'Chevrolet'
        })
        # 还原原始 NaN 位置，防止 "nan" 字符串被当作合法品牌
        result[brand_series.isna()] = np.nan
        return result

    def _extract_engine(self, engine_series):
        """多级正则自适应文本特征抽取（解决乱码与异构文本描述）"""
        hp_list = []
        liter_list = []
        for text in engine_series.astype(str):
            if pd.isna(text) or text.lower() == 'nan':
                hp_list.append(np.nan)
                liter_list.append(np.nan)
                continue

            # 第一级：提取马力 (HP)
            hp_match = re.search(r'(\d+\.?\d*)\s*HP', text, re.IGNORECASE)
            hp_list.append(float(hp_match.group(1)) if hp_match else np.nan)

            # 第二级：提取排量 (L / Liter)
            liter_match = re.search(r'(\d+\.?\d*)\s*(L|Liter)', text, re.IGNORECASE)
            if liter_match:
                liter_list.append(float(liter_match.group(1)))
            else:
                # 备用正则：捕捉缺失"L"但含有发动机缸体描述的样本（如 3.5 V6）
                backup_match = re.search(r'(\d+\.\d+)\s+V\d', text, re.IGNORECASE)
                liter_list.append(float(backup_match.group(1)) if backup_match else np.nan)

        return pd.Series(hp_list, index=engine_series.index), pd.Series(liter_list, index=engine_series.index)

    def fit(self, X, y=None):
        X_copy = X.copy()

        # 1. 学习非结构化引擎特征统计量（中位数）
        hp, liter = self._extract_engine(X_copy['engine'])
        self.hp_median_ = hp.median()
        self.liter_median_ = liter.median()

        # 极端噪声兜底保护：融合全球民用车保有量最大核心参数（2.0L / 200HP）
        if pd.isna(self.hp_median_): self.hp_median_ = 200.0
        if pd.isna(self.liter_median_): self.liter_median_ = 2.0

        # 2. 建立高频品牌字典，隔离测试集未登录词
        standard_brands = self._standardize_brands(X_copy['brand'])
        brand_counts = standard_brands.value_counts()
        self.frequent_brands_ = brand_counts[brand_counts >= self.rare_brand_threshold].index.tolist()

        # 3. 按品牌局部学习里程(milage)的局部IQR截断边界，严格杜绝数据窥探
        X_copy['brand_clean'] = standard_brands.apply(lambda x: x if x in self.frequent_brands_ else 'Other')
        for brand in X_copy['brand_clean'].unique():
            brand_milage = X_copy[X_copy['brand_clean'] == brand]['milage']
            if len(brand_milage) > 0:
                q1 = brand_milage.quantile(0.25)
                q3 = brand_milage.quantile(0.75)
                iqr = q3 - q1
                self.brand_iqr_bounds_[brand] = {
                    'lower': max(0, q1 - self.iqr_multiplier * iqr),
                    'upper': q3 + self.iqr_multiplier * iqr
                }
        return self

    def transform(self, X):
        X_out = X.copy()
        if 'id' in X_out.columns:
            X_out = X_out.drop(columns=['id'])

        X_out['brand'] = self._standardize_brands(X_out['brand'])
        X_out['brand'] = X_out['brand'].apply(lambda x: x if x in self.frequent_brands_ else 'Other')

        # 4. 亮点：温索尔自适应局部截断，应对网页端录入极端脏数据，保障服务永不崩溃
        for brand in self.brand_iqr_bounds_:
            idx = X_out['brand'] == brand
            if idx.any():
                X_out.loc[idx, 'milage'] = X_out.loc[idx, 'milage'].clip(
                    lower=self.brand_iqr_bounds_[brand]['lower'],
                    upper=self.brand_iqr_bounds_[brand]['upper']
                )

        # 5. 业务联动填充逻辑（MAR机制）
        tesla_idx = (X_out['brand'] == 'Tesla') & (X_out['fuel_type'].isna())
        X_out.loc[tesla_idx, 'fuel_type'] = 'Electric'
        elec_engine_idx = (X_out['engine'].str.contains('Electric', case=False, na=False)) & (X_out['fuel_type'].isna())
        X_out.loc[elec_engine_idx, 'fuel_type'] = 'Electric'
        X_out['fuel_type'] = X_out['fuel_type'].fillna('Unknown')

        # 6. 提取引擎数值并安全填充
        hp, liter = self._extract_engine(X_out['engine'])
        X_out['engine_hp'] = hp.fillna(self.hp_median_)
        X_out['engine_liter'] = liter.fillna(self.liter_median_)

        # 7. 亮点：车况缺失提炼为独立新类别 "Unknown"，交由算法学习隐藏欺诈权重
        if 'accident' in X_out.columns:
            def clean_accident(x):
                if pd.isna(x): return 'Unknown'
                elif 'at least 1 accident' in str(x).lower(): return 'Has_Accident'
                else: return 'No_Accident'
            X_out['accident_status'] = X_out['accident'].apply(clean_accident)
            X_out = X_out.drop(columns=['accident'])

        # 8. 产权特征转换
        if 'clean_title' in X_out.columns:
            X_out['is_clean_title'] = X_out['clean_title'].apply(
                lambda x: 1 if pd.notna(x) and str(x).lower() == 'yes' else 0)
            X_out = X_out.drop(columns=['clean_title'])

        # 9. 时间物理特征抽取
        if 'model_year' in X_out.columns:
            X_out['car_age'] = CURRENT_YEAR - X_out['model_year']

        # 10. 特征裁剪：保留 model 列（高价值车型信息，后续做 Target Encoding）
        drop_cols = ['engine', 'model_year', 'transmission', 'ext_col', 'int_col']
        X_out = X_out.drop(columns=[c for c in drop_cols if c in X_out.columns])

        return X_out