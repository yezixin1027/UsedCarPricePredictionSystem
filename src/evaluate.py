# src/evaluate.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, ACTIVE_MODEL
from src.preprocess import AdvancedUsedCarPreprocessor
from src.features import HighScoreFeatureEngineer
from src.models import UsedCarModelFactory


def run_advanced_audit():
    print("=" * 70)
    print(f"[高级审计流水线启动] 当前选定审计核心模型: {ACTIVE_MODEL.upper()}")
    print("=" * 70)

    # 1. 载入原始完整大盘资产
    raw_df = pd.read_csv(TRAIN_PATH)
    X_raw = raw_df.drop(columns=['price'])
    y_true_real = raw_df['price'].values
    y_log = np.log1p(y_true_real)

    # 初始化 OOF 容器
    oof_preds_real = np.zeros(len(raw_df))
    model = UsedCarModelFactory.create_model(ACTIVE_MODEL)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_raw):
        X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train = y_log[train_idx]

        # 管道隔离拟合
        preprocessor = AdvancedUsedCarPreprocessor()
        fe_engineer = HighScoreFeatureEngineer()
        X_train_feat = fe_engineer.fit_transform(preprocessor.fit_transform(X_train_raw, y_train), y_train)
        X_val_feat = fe_engineer.transform(preprocessor.transform(X_val_raw))

        model.fit(X_train_feat.values, y_train)
        preds_log = model.predict(X_val_feat.values)

        # 🔥 【核心关键】：逆对数转换，必须把预测结果还原到真实的“车价账面金额空间”
        oof_preds_real[val_idx] = np.expm1(preds_log)

    # 构建综合审计大表
    audit_df = raw_df.copy()
    audit_df['predicted_price'] = oof_preds_real
    audit_df['absolute_error_real'] = np.abs(audit_df['price'] - audit_df['predicted_price'])
    audit_df['residual_real'] = audit_df['price'] - audit_df['predicted_price']  # 正代表低估，负代表高估
    audit_df['mape_real'] = audit_df['absolute_error_real'] / audit_df['price']

    # ============================================================
    # 审计点 ①：基础业务指标统计 (人民币/美元 真实金额空间)
    # ============================================================
    global_mae = mean_absolute_error(y_true_real, oof_preds_real)
    global_rmse = np.sqrt(mean_squared_error(y_true_real, oof_preds_real))
    global_mape = audit_df['mape_real'].mean()
    global_r2 = r2_score(y_log, np.log1p(oof_preds_real))  # 记录对数空间R2

    print(f"\n[📊 审计报告1 · 全局多维度学术指标]")
    print(f"  -> 全局真实车价空间 MAE (平均绝对误差):  {global_mae:.2f} 元")
    print(f"  -> 全局真实车价空间 RMSE (均方根误差):  {global_rmse:.2f} 元")
    print(f"  -> 全局真实车价空间 MAPE (平均百分误差): {global_mape * 100:.2f}%")
    print(f"  -> 最终沉淀对数空间 5折 CV R² 决定系数: {global_r2:.4f}")

    # ============================================================
    # 审计点 ②：黑盒残差审计 —— 追踪极端误差 Top-10 样本
    # ============================================================
    print(f"\n[🔍 审计报告2 · 极端看走眼大误差 Top-10 追踪]")
    top10_error = audit_df.sort_values(by='absolute_error_real', ascending=False).head(10)

    # 打印核心关键字段进行肉眼特征分析
    print(top10_error[
              ['brand', 'model', 'model_year', 'milage', 'engine', 'price', 'predicted_price', 'absolute_error_real']])
    # 固化为 Excel，作为论文第4章第4节深度个案剖析的铁证
    os.makedirs('reports', exist_ok=True)
    top10_error.to_excel('reports/top10_extreme_errors.xlsx', index=False)
    print("  结论: 极端误差样本详细报表已导出至 -> reports/top10_extreme_errors.xlsx")

    # ============================================================
    # 审计点 ③：可信任机器学习 —— 系统性公平性与算法偏见评估
    # ============================================================
    print(f"\n[⚖️ 审计报告3 · 价格区间群体算法公平性评估]")
    # 基于行业常识对真实车价进行分箱
    bins = [0, 15000, 35000, 75000, np.inf]
    labels = ['低端代步车(<1.5万)', '中端实用车(1.5-3.5万)', '高档轻奢车(3.5-7.5万)', '顶级豪华超跑(>7.5万)']
    audit_df['price_group'] = pd.cut(audit_df['price'], bins=bins, labels=labels)

    fairness_price = audit_df.groupby('price_group', observed=False).agg(
        样本量=('id', 'count'),
        平均绝对误差MAE=('absolute_error_real', 'mean'),
        平均残差MeanResidual=('residual_real', 'mean'),
        组内百分比误差MAPE=('mape_real', 'mean')
    )
    print(fairness_price)

    print(f"\n[⚖️ 审计报告4 · 品牌保值率流派算法公平性评估 (前5大主流品牌)]")
    top_brands = audit_df['brand'].value_counts().head(5).index
    fairness_brand = audit_df[audit_df['brand'].isin(top_brands)].groupby('brand').agg(
        样本量=('id', 'count'),
        平均残差MeanResidual=('residual_real', 'mean'),
        组内百分比误差MAPE=('mape_real', 'mean')
    )
    print(fairness_brand)


if __name__ == "__main__":
    run_advanced_audit()


# ================================================================
# 扩展评估函数 (供 run_training.py 调用)
# ================================================================

def analyze_top_errors(audit_df, top_n=10, save_path=None):
    """分析预测误差最大的 Top-N 样本，总结共性规律。

    Parameters
    ----------
    audit_df : pd.DataFrame
        需包含: predicted_price, price, absolute_error_real, residual_real, mape_real
    top_n : int
    save_path : str or None
        若提供则导出 Excel

    Returns
    -------
    pd.DataFrame
        Top-N 误差样本
    """
    top_errors = audit_df.sort_values(
        'absolute_error_real', ascending=False).head(top_n)

    print(f"\n  {'─' * 80}")
    print(f"  Top-{top_n} 极端误差样本")
    print(f"  {'─' * 80}")
    for i, (idx, row) in enumerate(top_errors.iterrows(), 1):
        direction = '低估' if row['residual_real'] > 0 else '高估'
        print(f"  #{i}: 真实={row['price']:,.0f} | 预测={row['predicted_price']:,.0f} | "
              f"误差={row['absolute_error_real']:,.0f} ({row['mape_real']*100:.1f}%) | "
              f"{direction}")

    # 共性分析
    print(f"\n  [共性分析]")
    # 高估 vs 低估
    overestimate = (top_errors['residual_real'] < 0).sum()
    underestimate = (top_errors['residual_real'] > 0).sum()
    print(f"    高估 (预测 > 真实): {overestimate}/{top_n} 条")
    print(f"    低估 (预测 < 真实): {underestimate}/{top_n} 条")

    # 价格区间分布
    bins = [0, 15000, 35000, 75000, np.inf]
    labels = ['低端 (<1.5万)', '中端 (1.5-3.5万)', '高档 (3.5-7.5万)', '豪华 (>7.5万)']
    top_errors_prices = top_errors['price'].values
    price_dist = pd.cut(top_errors_prices, bins=bins, labels=labels).value_counts()
    print(f"    价格区间分布:")
    for label, count in price_dist.items():
        print(f"      {label}: {count}/{top_n}")

    # 平均误差
    avg_error = top_errors['absolute_error_real'].mean()
    avg_price_top = top_errors['price'].mean()
    print(f"    极端样本均价: {avg_price_top:,.0f} | 平均误差: {avg_error:,.0f} "
          f"({avg_error/avg_price_top*100:.1f}%)")
    print(f"    改进建议: 极端误差集中在 {'高价' if avg_price_top > 35000 else '中低价'} "
          f"区间，{'可能需要更多高端车训练样本' if avg_price_top > 35000 else '需关注低端车的数据质量'}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'reports',
                    exist_ok=True)
        top_errors.to_excel(save_path, index=False)
        print(f"    [OK] 极端误差样本已导出: {save_path}")

    return top_errors


def run_fairness_audit(audit_df, group_col='price_group'):
    """对指定分组列执行公平性审计，检测系统性偏差。

    Parameters
    ----------
    audit_df : pd.DataFrame
        需包含: price_group (或 group_col), absolute_error_real, residual_real, mape_real
    group_col : str
        分组列名

    Returns
    -------
    pd.DataFrame
        分组聚合结果: count, MAE, mean_residual, MAPE
    """
    agg = audit_df.groupby(group_col, observed=False).agg(
        count=('price', 'count'),
        MAE=('absolute_error_real', 'mean'),
        mean_residual=('residual_real', 'mean'),
        MAPE=('mape_real', 'mean')
    )

    # 检测系统性偏差
    global_mae = audit_df['absolute_error_real'].mean()
    for idx, row in agg.iterrows():
        bias = row['mean_residual']
        if abs(bias) > global_mae * 0.2:
            direction = '被系统性低估' if bias > 0 else '被系统性高估'
            print(f"    ⚠ [{idx}] 存在系统性偏差: 均残差={bias:,.0f}, {direction}")

    return agg