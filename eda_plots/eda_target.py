# eda_plots/eda_target.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_PATH, FIGURES_DIR

# 绘制原始与对数转换对比图
train_df = pd.read_csv(TRAIN_PATH)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['price'], kde=True, bins=50)
plt.title('Original Price Distribution (Right-skewed)')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df['price']), kde=True, bins=50, color='green')
plt.title('Log-transformed Price Distribution (Normal)')

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, 'price_distribution_comparison.png'), dpi=300)
print("目标变量分布对比图生成完毕！")
plt.show()
