# src/selection.py
"""
特征选择与降维模块
===================

第一轮 · VIF 多重共线性剔除
    计算每个连续数值特征的方差膨胀因子 (Variance Inflation Factor, VIF)。
    VIF_j = 1 / (1 - R²_j)，其中 R²_j 为第 j 个特征对其他所有特征的回归决定系数。
    经验阈值 VIF > 10 判定为严重共线性，迭代剔除 VIF 最高的特征直至全部达标。
    筛选标准: 高 VIF 导致 OLS 系数方差膨胀 VIF 倍 → 模型解释性下降 → 剔除冗余。

第二轮 · 互信息 (Mutual Information) 重要性筛选
    基于信息论的非参数相关性度量，能捕获线性和非线性依赖。
    对 VIF 筛选后的所有特征计算与目标变量的 MI 值，
    保留 MI > 自适应阈值（所有特征 MI 均值的 5%）的特征。
    筛选标准: MI 过低 → 特征对目标预测贡献极微 → 降维以减少过拟合风险。

输出:
    - 特征筛选详细报告 (console log)
    - MI 重要性排序条形图 (reports/figures/feature_mi_ranking.png)


"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端，兼容服务器/CI 环境
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression

# 学术图表配置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FIGURES_DIR


def _calculate_vif(X_numeric):
    """计算数据框中各数值列的 VIF。

    使用相关系数矩阵逆的对角元方法:
        VIF_j = (R^{-1})_{jj}
    其中 R 是特征间的皮尔逊相关系数矩阵。
    该方法等价于 1 / (1 - R²_j)，但数值更稳定。

    Parameters
    ----------
    X_numeric : pd.DataFrame
        仅含数值列的 DataFrame (标准化后效果最佳)。

    Returns
    -------
    pd.Series : 列名 → VIF 值
    """
    cols = X_numeric.columns.tolist()
    if len(cols) < 2:
        return pd.Series({cols[0]: 1.0} if cols else {})

    # 计算相关系数矩阵
    corr = X_numeric.corr().values
    # 对近似奇异矩阵添加微小正则化以保证逆存在
    try:
        corr_inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        corr_inv = np.linalg.pinv(corr)
    vif_vals = np.diag(corr_inv)
    # 确保非负 (理论上 VIF >= 1)
    vif_vals = np.maximum(vif_vals, 1.0)
    return pd.Series(vif_vals, index=cols)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """两阶段特征选择器。

    阶段 1 — VIF 多重共线性筛选: 迭代剔除 VIF > 10 的特征。
    阶段 2 — 互信息重要性筛选: 剔除 MI < 自适应阈值的弱相关特征。

    Parameters
    ----------
    vif_threshold : float
        VIF 剔除阈值 (default=10.0)。VIF > 10 通常判定为严重共线性。
    mi_percentile_factor : float
        MI 阈值为所有特征 MI 均值乘以此因子 (default=0.05, 即均值的 5%)。
    random_state : int
        随机种子，用于 MI 计算的可重复性 (default=42)。
    verbose : bool
        是否打印筛选日志 (default=True)。
    """
    def __init__(self, vif_threshold=10.0, mi_percentile_factor=0.05,
                 random_state=42, verbose=True):
        self.vif_threshold = vif_threshold
        self.mi_percentile_factor = mi_percentile_factor
        self.random_state = random_state
        self.verbose = verbose
        # fit 后填充
        self.selected_features_ = None       # 最终保留的特征列表
        self.removed_by_vif_ = []            # VIF 剔除的特征
        self.removed_by_mi_ = []             # MI 剔除的特征
        self.vif_history_ = {}               # 各迭代轮次的 VIF 值
        self.mi_scores_ = None               # 最终 MI 分值
        self.removal_report_ = []            # 详细剔除报告

    def fit(self, X, y):
        """在训练集上执行两阶段特征筛选。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵（可能含 One-Hot 编码后的分类列）。
        y : array-like
            原始目标变量（price，未做对数变换）。内部自动 log1p。

        Returns
        -------
        self
        """
        X_copy = X.copy()
        y_log = np.log1p(y)

        if self.verbose:
            self._log("=" * 70)
            self._log("  特征选择器 · 两阶段筛选启动")
            self._log(f"  输入特征维度: {X_copy.shape[1]} 列")
            self._log("=" * 70)

        # ---- 分离数值列与二值列 ----
        numeric_cols = []
        binary_cols = []
        for col in X_copy.columns:
            unique_vals = X_copy[col].dropna().unique()
            if len(unique_vals) <= 2:
                binary_cols.append(col)
            elif np.issubdtype(X_copy[col].dtype, np.number):
                numeric_cols.append(col)
            else:
                binary_cols.append(col)  # 字符串类视为已编码分类

        if self.verbose:
            self._log(f"\n  特征分类: 数值 {len(numeric_cols)} 列 | "
                      f"二值/分类 {len(binary_cols)} 列")

        # ================================================================
        # 第一轮: VIF 多重共线性筛选 (仅对数值特征)
        # ================================================================
        if self.verbose:
            self._log(f"\n  {'─' * 60}")
            self._log(f"  [第一轮] VIF 多重共线性筛选 (阈值 VIF > {self.vif_threshold})")
            self._log(f"  {'─' * 60}")

        current_numeric = numeric_cols.copy()
        self.vif_history_ = {}
        round_num = 0

        while len(current_numeric) >= 2:
            round_num += 1
            vif_series = _calculate_vif(X_copy[current_numeric])
            self.vif_history_[f"round_{round_num}"] = vif_series.to_dict()
            max_vif_col = vif_series.idxmax()
            max_vif_val = vif_series.max()

            if self.verbose:
                status = "[OK] 全部达标" if max_vif_val <= self.vif_threshold else "[X] 超标"
                self._log(f"  迭代 {round_num}: max VIF = {max_vif_val:8.2f} "
                          f"({max_vif_col:<20}) {status}")

            if max_vif_val <= self.vif_threshold:
                break

            # 剔除 VIF 最高的特征
            reason = (f"VIF={max_vif_val:.1f} (远超阈值{self.vif_threshold}), "
                      f"与其他特征高度线性依赖, 导致协方差矩阵不稳定")
            self.removal_report_.append({
                'feature': max_vif_col,
                'round': 'VIF',
                'metric': f'VIF = {max_vif_val:.2f}',
                'reason': reason
            })
            self.removed_by_vif_.append(max_vif_col)
            current_numeric.remove(max_vif_col)

        if len(current_numeric) <= 1 and round_num == 0:
            self._log("  (数值特征不足2列, 跳过VIF筛选)")

        # ================================================================
        # 第二轮: 互信息 (MI) 重要性筛选 (面向所有候选特征)
        # ================================================================
        candidate_cols = current_numeric + binary_cols

        if self.verbose:
            self._log(f"\n  {'─' * 60}")
            self._log(f"  [第二轮] 互信息 (Mutual Information) 重要性筛选")
            self._log(f"  候选特征: {len(candidate_cols)} 列")
            self._log(f"  {'─' * 60}")

        # 准备 MI 计算矩阵（确保全数值）
        X_mi = X_copy[candidate_cols].astype(float).fillna(0).values

        mi_raw = mutual_info_regression(
            X_mi, y_log,
            discrete_features='auto',
            random_state=self.random_state,
            n_neighbors=3
        )
        self.mi_scores_ = pd.Series(mi_raw, index=candidate_cols).sort_values(ascending=False)

        # 自适应阈值: 所有特征 MI 均值的 mi_percentile_factor 倍
        mi_threshold = self.mi_scores_.mean() * self.mi_percentile_factor

        if self.verbose:
            self._log(f"  MI 范围: [{self.mi_scores_.min():.6f}, "
                      f"{self.mi_scores_.max():.4f}]")
            self._log(f"  MI 均值: {self.mi_scores_.mean():.6f}")
            self._log(f"  自适应阈值 (均值 × {self.mi_percentile_factor}): {mi_threshold:.6f}")
            self._log(f"\n  {'特征':<30} {'MI值':>10} {'阈值判定':>12}")
            self._log(f"  {'─' * 55}")

        for feat, mi_val in self.mi_scores_.items():
            if mi_val < mi_threshold:
                reason = (f"MI={mi_val:.6f} < 阈值{mi_threshold:.6f}, "
                          f"与对数价格的统计依赖极弱, 对预测贡献可忽略")
                self.removal_report_.append({
                    'feature': feat,
                    'round': 'MI',
                    'metric': f'MI = {mi_val:.6f}',
                    'reason': reason
                })
                self.removed_by_mi_.append(feat)
                status = "[X] 剔除"
            else:
                status = "[OK] 保留"
            if self.verbose:
                self._log(f"  {feat:<30} {mi_val:>10.6f} {status:>12}")

        # 确定最终保留特征
        all_removed = set(self.removed_by_vif_ + self.removed_by_mi_)
        self.selected_features_ = [c for c in candidate_cols if c not in all_removed]

        if self.verbose:
            self._log(f"\n  {'=' * 60}")
            self._log(f"  筛选完成:")
            self._log(f"    原始特征: {X_copy.shape[1]} 列")
            self._log(f"    VIF 剔除: {len(self.removed_by_vif_)} 列")
            self._log(f"    MI  剔除: {len(self.removed_by_mi_)} 列")
            self._log(f"    最终保留: {len(self.selected_features_)} 列")
            self._log(f"    降维比例: {(1 - len(self.selected_features_) / X_copy.shape[1]) * 100:.1f}%")
            self._log(f"  {'=' * 60}")

        # ---- 生成 MI 排序图 ----
        self._plot_mi_ranking(mi_threshold)

        return self

    def transform(self, X):
        """按筛选结果裁剪测试集特征列。

        Parameters
        ----------
        X : pd.DataFrame
            待裁剪的特征矩阵。

        Returns
        -------
        pd.DataFrame
            仅保留 selected_features_ 列的特征矩阵。
        """
        if self.selected_features_ is None:
            raise RuntimeError("FeatureSelector 尚未 fit, 请先调用 fit()")
        # 容错：部分列可能因 One-Hot 处理而命名不一致
        available = [c for c in self.selected_features_ if c in X.columns]
        missing = set(self.selected_features_) - set(available)
        if missing and self.verbose:
            self._log(f"  [FeatureSelector.transform] 警告: "
                      f"测试集中缺少以下列: {missing}, 已自动跳过")
        return X[available].copy()

    def get_report(self):
        """以 DataFrame 格式返回筛选报告。

        Returns
        -------
        pd.DataFrame
            列: ['feature', 'round', 'metric', 'reason']
        """
        return pd.DataFrame(self.removal_report_)

    def _plot_mi_ranking(self, mi_threshold):
        """绘制互信息重要性排序条形图 (毕业设计规范级)。"""
        os.makedirs(FIGURES_DIR, exist_ok=True)
        scores = self.mi_scores_
        n_features = len(scores)

        fig, ax = plt.subplots(figsize=(10, max(5, n_features * 0.35)))

        # 按保留/剔除着色
        bar_colors = [
            '#2F5496' if feat not in self.removed_by_mi_ else '#C00000'
            for feat in scores.index
        ]

        bars = ax.barh(range(n_features), scores.values, color=bar_colors,
                       edgecolor='white', linewidth=0.8)

        # 标注 MI 值
        for i, (feat, val) in enumerate(zip(scores.index, scores.values)):
            ax.text(val + max(scores.values) * 0.01, i, f'{val:.4f}',
                    va='center', fontsize=8)

        # 阈值线
        ax.axvline(x=mi_threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'自适应阈值 = {mi_threshold:.6f}')

        ax.set_yticks(range(n_features))
        ax.set_yticklabels(scores.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('互信息 (Mutual Information)', fontsize=11)
        ax.set_title('特征互信息 (MI) 重要性排序\n'
                     '蓝色 = 保留 | 红色 = MI 值低于阈值被剔除',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)

        # 颜色图例
        legend_elements = [
            Patch(facecolor='#2F5496', label=f'保留特征 ({n_features - len(self.removed_by_mi_)})'),
            Patch(facecolor='#C00000', label=f'MI 剔除 ({len(self.removed_by_mi_)})'),
        ]
        ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red',
                   linestyle='--', label=f'阈值 = {mi_threshold:.6f}')],
                  loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_mi_ranking.png'), dpi=300)
        plt.close()  # 释放图形资源，避免阻塞
        if self.verbose:
            self._log(f"\n  [OK] MI 排序图已保存: reports/figures/feature_mi_ranking.png")

    @staticmethod
    def _log(msg):
        # Windows GBK 编码安全输出
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('gbk', errors='replace').decode('gbk'))
