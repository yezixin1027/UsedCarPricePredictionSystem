# 二手车价格预测系统 · Used Car Price Prediction System

基于 18.8 万条真实二手车交易数据的**工业级价格预测系统**。

---

## 项目结构与论文章节映射

```
UsedCarPricePredictionSystem/
│
├── run_all.py                       # 🔥 一键运行全流程
├── config.py                        # 全局路径/超参数/常量
├── README.md
│
├── exploration/                     # 📄 第2章 数据探索性分析
│   ├── run_exploration.py           #    阶段入口
│   ├── s1_dataset_overview.py       #    2.1 数据集概览
│   ├── s2_target_analysis.py        #    2.2 目标变量分析
│   ├── s3_correlation_heatmap.py    #    2.3 特征相关性分析
│   ├── s4_feature_transform.py      #    2.4 数值特征变换对比
│   └── s5_feature_validation.py     #    2.5 衍生特征有效性验证
│
├── preprocessing/                   # 📄 第3章 数据预处理与特征工程
│   ├── run_preprocessing.py         #    阶段入口 (生成特征矩阵)
│   ├── s1_data_cleaning.py          #    3.1 数据清洗与异常值处理
│   ├── s2_encoding_comparison.py    #    3.2 分类特征编码方式对比
│   ├── s3_feature_engineering.py    #    3.3 特征工程
│   └── s4_feature_selection.py      #    3.4 特征选择与降维
│
├── modeling/                        # 📄 第4章 模型构建与评估
│   ├── run_modeling.py              #    阶段入口 (训练+对比+保存)
│   ├── s1_model_training.py         #    4.1-4.2 模型选择与训练
│   ├── s2_hyperparameter_tuning.py  #    4.2 超参数调优
│   ├── s3_model_comparison.py       #    4.3 模型性能对比分析
│   └── s4_model_explainability.py   #    4.4 最佳模型深度分析
│
├── analysis/                        # 📄 第5章 结果分析与决策优化
│   ├── run_analysis.py              #    阶段入口
│   ├── s1_prediction_analysis.py    #    5.1 预测结果分析
│   └── s2_business_insights.py      #    5.2 业务建议
│
├── application/                     # 📄 第6章 系统设计与实现
│   ├── app.py                       #    Streamlit 估价界面
│   └── run_app.py                   #    启动脚本
│
├── src/                             # 🔧 核心模块库
│   ├── preprocess.py                #    数据清洗类
│   ├── features.py                  #    特征工程类 (4衍生+1交互+编码+标准化)
│   ├── selection.py                 #    特征选择类 (VIF+MI)
│   ├── models.py                    #    模型工厂 (5模型+Stacking)
│   ├── train.py                     #    训练器+分层CV+雷达图
│   ├── tune_all_models.py           #    Optuna 超参数调优
│   ├── evaluate.py                  #    高级审计 (Top10误差+公平性)
│   └── explainability.py            #    SHAP 可解释性分析
│
├── data/  /  models/  /  reports/   # 数据 / 模型 / 报告图表
└── eda_plots/  / app/               # (保留兼容, 内容已迁移到新目录)
```

---

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 放入数据文件到 data/raw/

# 3. 一键运行全流程
python run_all.py

# 或分步运行:
python exploration/run_exploration.py      # 第2章: 数据探索
python preprocessing/run_preprocessing.py  # 第3章: 预处理+特征工程
python modeling/run_modeling.py            # 第4章: 模型训练+对比
python analysis/run_analysis.py            # 第5章: 结果分析

# 4. 启动 Web 界面
python application/run_app.py
# 或: streamlit run application/app.py
```

---

## 核心亮点

### 数据预处理 (第3章)
- 品牌名称标准化 + 稀有品牌归拢（<15→Other）
- **品牌级局部 IQR 截尾**（按品牌分组，避免豪车被全局阈值误杀）
- MAR 机制缺失值业务联动填充
- 3种编码方式定量对比实验（Drop vs One-Hot vs Target Encoding）

### 特征工程 (第3章)
- **4个经验证有效的衍生特征**：annual_milage, power_density, car_age_squared, hp_per_year
- **1个交互特征**：brand_x_milage
- 目标编码(OOF) + One-Hot + Z-score 标准化
- VIF + MI 两阶段特征筛选（~26→11维，降维57.7%）

### 模型训练 (第4章)
| 模型 | R² | MAE (元) |
|------|------|-----------|
| XGBoost | 0.6626 | 17,170 |
| LightGBM | 0.6622 | 17,189 |
| CatBoost | 0.6593 | 17,261 |
| Random Forest | 0.6558 | 17,311 |
| Ridge | 0.6189 | 18,189 |

### 模型评估 (第4-5章)
- 多维度雷达图对比
- Top-10 极端误差样本分析
- 品牌/价格区间公平性审计
- SHAP 特征重要性 + 依赖图 + 瀑布图

---

## 特征清单

| 特征 | 类型 | 筛选结果 | XGBoost重要性 |
|------|------|----------|--------------|
| hp_per_year | 衍生 | ✅ 保留 | 51.8% (#1) |
| milage | 基础 | ✅ 保留 | 17.8% (#2) |
| car_age_squared | 衍生 | ✅ 保留 | 13.6% (#3) |
| model_encoded | 编码 | ✅ 保留 | 5.3% (#4) |
| engine_hp | 基础 | ✅ 保留 | 2.2% (#5) |
| annual_milage | 衍生 | ✅ 保留 | - |
| power_density | 衍生 | ✅ 保留 | - |
| brand_encoded | 编码 | ✅ 保留 | - |
| brand_x_milage | 交互 | ✅ 保留 | - |
| milage_log | 衍生 | ❌ VIF剔除 | - |
| engine_torque_proxy | 衍生 | ❌ VIF剔除 | - |
| brand_x_car_age | 交互 | ❌ MI剔除 | - |
| brand_x_power | 交互 | ❌ MI剔除 | - |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn, LightGBM, XGBoost, CatBoost |
| 超参数优化 | Optuna (TPE 贝叶斯优化) |
| 可解释性 | SHAP (TreeExplainer) |
| 可视化 | Matplotlib, Seaborn |
| Web 界面 | Streamlit |
| 包管理 | uv |
