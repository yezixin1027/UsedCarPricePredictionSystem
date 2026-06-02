# 二手车价格预测系统 · Used Car Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6+-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-red.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.2+-blueviolet.svg)](https://optuna.org/)

基于 18.8 万条真实二手车交易数据的**工业级价格预测系统**，覆盖从数据清洗、高维特征工程、特征选择、五模型训练对比、超参数自动调优、Stacking 集成融合到 SHAP 可解释性分析的完整机器学习流水线。

---

## 核心亮点

### 防数据泄露
- **五折嵌套隔离目标编码（OOF Target Encoding）**：对 brand 和 model 两个高基数分类变量做拉普拉斯平滑目标编码，严格在训练折内计算
- **fit/transform 分离**：所有预处理器和特征工程器继承 `sklearn` 标准接口

### 自适应数据清洗
- 品牌名称标准化 + 稀有品牌归拢（<15 样本 → Other）
- 正则引擎特征抽取（从非结构化文本提取 HP 和排量）
- **品牌级局部 IQR 截尾**（按品牌分组，避免豪车被全局阈值"误杀"）
- MAR 机制缺失值业务联动填充（Tesla → Electric 等）

### 创新特征体系
- **6 个衍生特征**：annual_milage, power_density, car_age_squared, milage_log, hp_per_year, engine_torque_proxy
- **3 个交互特征**：brand_x_car_age, brand_x_milage, brand_x_power
- 26 个候选特征 → **11 个精选特征**（VIF + MI 两阶段筛选，降维 57.7%）

### 五模型 + Stacking 集成

| 模型 | 类型 | R² | MAE (元) |
|------|------|-----|-----------|
| XGBoost | Boosting (预排序) | 0.6626 | 17,170 |
| LightGBM | Boosting (直方图) | 0.6622 | 17,189 |
| CatBoost | Boosting (有序提升) | 0.6593 | 17,261 |
| Random Forest | Bagging | 0.6558 | 17,311 |
| Ridge | 线性 (L2) | 0.6189 | 18,189 |

---

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 放入数据文件
# 将 train.csv 和 test.csv 放入 data/raw/

# 3. 运行数据流水线（预处理 → 特征工程 → 特征选择 → 持久化）
python run_pipeline.py

# 4. 训练全部模型 + Stacking + 保存权重
python run_training.py --predict

# 5. 超参数调优（可选，30-60 分钟）
python src/tune_all_models.py --trials 30

# 6. SHAP 可解释性分析
python src/explainability.py

# 7. 编码方式对比实验
python src/encoding_compare.py

# 8. Web 演示界面
streamlit run app/app.py
```

---

## 项目结构

```
UsedCarPricePredictionSystem/
│
├── run_pipeline.py                # [入口] 数据预处理流水线
├── run_training.py                # [入口] 五模型训练 + Stacking + 保存权重
├── README.md                      # 项目说明文档
├── pyproject.toml                 # 依赖管理 (uv)
│
├── src/                           # 核心源码（9 个模块）
│   ├── config.py                  # 全局路径 & 模型参数配置
│   ├── preprocess.py              # 数据预处理（品牌清洗/引擎抽取/IQR截尾/MAR填充）
│   ├── features.py                # 特征工程（6衍生+3交互+2编码+标准化）
│   ├── selection.py               # 特征选择（VIF共线性 + MI互信息）
│   ├── models.py                  # 模型工厂（5模型 + Stacking）
│   ├── train.py                   # 训练器 + 分层CV + 雷达图 + 参数曲线
│   ├── evaluate.py                # 高级审计（Top10误差 + 公平性）
│   ├── tune_all_models.py         # 五模型统一 Optuna 调优
│   ├── explainability.py          # SHAP 可解释性分析
│   └── encoding_compare.py        # 编码方式对比实验
│
├── eda_plots/                     # 探索性数据分析（4 个脚本）
│   ├── eda_target.py              # 目标变量分布对比
│   ├── eda_heatmap.py             # 特征相关性热力图
│   ├── eda_feature_transform.py   # 特征变换策略对比
│   └── eda_features_validation.py # 衍生特征有效性验证（5维度）
│
├── app/                           # Web 演示
│   └── app.py                     # Streamlit 交互式估价界面
│
├── data/                          # 数据（不纳入版本控制）
│   ├── raw/                       # 原始 CSV（需自行放入）
│   └── processed/                 # 处理后的特征矩阵（自动生成）
│
├── models/                        # 模型权重（自动生成）
│   ├── ridge_model.pkl
│   ├── random_forest_model.pkl
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── catboost_model.pkl
│   ├── model_comparison.csv       # 性能对比表
│   └── feature_importance_summary.csv
│
└── reports/                       # 报告与图表
    ├── 项目报告（模版）.docx        # 项目报告模版
    ├── figures/                   # 15+ 可视化图表
    └── top10_extreme_errors.xlsx  # 极端误差样本
```

---

## 各文件作用说明

| 文件 | 作用 | 运行方式 |
|------|------|----------|
| `run_pipeline.py` | 原始数据 → 预处理 → 特征工程 → 特征选择 → 持久化 | `python run_pipeline.py` |
| `run_training.py` | 加载特征矩阵 → 五模型训练 → Stacking → 保存权重 → 评估 | `python run_training.py` |
| `src/tune_all_models.py` | Ridge/RF 网格搜索 + LGB/XGB/Cat Optuna 贝叶斯优化 → 更新 config.py | `python src/tune_all_models.py --trials 30` |
| `src/explainability.py` | SHAP 可解释性分析（Summary/Importance/Dependence/Waterfall 4图） | `python src/explainability.py` |
| `src/encoding_compare.py` | 编码方式定量对比实验（Drop vs One-Hot vs Target Encoding） | `python src/encoding_compare.py` |
| `src/evaluate.py` | 独立高级审计（Top10误差 + 品牌/价格区间公平性） | `python src/evaluate.py` |
| `app/app.py` | Streamlit Web 交互式估价界面 | `streamlit run app/app.py` |
| `eda_plots/eda_*.py` | 探索性数据分析（分布/相关性/变换/验证） | `python eda_plots/eda_*.py` |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn, LightGBM, XGBoost, CatBoost |
| 超参数优化 | Optuna (TPE 贝叶斯优化) |
| 可解释性 | SHAP (TreeExplainer) |
| 可视化 | Matplotlib, Seaborn |
| Web 界面 | Streamlit |
| 包管理 | uv |

---

## 数据流

```
data/raw/train.csv (188,533 × 13)
       │
       ▼  src/preprocess.py  (品牌清洗/引擎抽取/IQR截尾/MAR填充)
       │
  (188,533 × 9)
       │
       ▼  src/features.py  (6衍生+3交互+2编码+标准化)
       │
  (188,533 × 26)
       │
       ▼  src/selection.py  (VIF剔除6 + MI剔除9)
       │
  (188,533 × 11)  →  data/processed/train_features.csv
       │
       ▼  run_training.py  (5折分层CV → 五模型训练 → Stacking)
       │
  models/*.pkl  (5个模型权重)
```
