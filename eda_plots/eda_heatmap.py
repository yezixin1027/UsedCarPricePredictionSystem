# eda_plots/eda_heatmap.py
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 优雅配置绘图风格与中文支持
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_PATH, FIGURES_DIR
from src.preprocess import AdvancedUsedCarPreprocessor

# 1. 导入原始训练集，应用高容错清洗
train_raw = pd.read_csv(TRAIN_PATH)
preprocessor = AdvancedUsedCarPreprocessor()

y_log = np.log1p(train_raw['price']).reset_index(drop=True)
X_clean = preprocessor.fit_transform(train_raw.drop(columns=['price']))

# 2. 🔥 【彻底解决 TypeError 核心修复点】：重置索引并显式强转浮点类型
X_clean = pd.DataFrame(X_clean).reset_index(drop=True)
target_cols = ['milage', 'engine_hp', 'engine_liter', 'car_age']

# 显式提取列、强转数据类型，彻底阻断由于 Pandas 局部索引冲突带来的类型隐式退化
analysis_df = X_clean[target_cols].astype(float).copy()
analysis_df['log_price'] = y_log

# 优化列名，满足毕业设计高可读性规范
analysis_df.columns = ['行驶里程', '发动机马力', '发动机排量', '车龄', '对数价格']

# 3. 计算皮尔逊相关系数并画图
corr_matrix = analysis_df.corr(method='pearson')

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # 顶刊标配上三角掩膜
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmax=1.0,
    vmin=-1.0,
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=.5,
    cbar_kws={"shrink": .7, "label": "皮尔逊相关系数 (r)"},
    square=True
)

plt.title("图 2-3 训练集核心连续特征与对数价格皮尔逊相关性矩阵热力图", fontsize=13, pad=20, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "feature_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
print("特征相关性热力图已生成！")
plt.show()