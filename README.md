# 二手车价格预测系统 · Used Car Price Prediction System

基于 18.8 万条真实二手车交易数据的**工业级机器学习价格预测系统**。

---

## 项目结构

```
UsedCarPricePredictionSystem/
│
├── run_all.py                       # 一键运行全流程
├── config.py                        # 全局路径 / 超参数 / 常量
├── README.md
│
├── exploration/                     # 数据探索性分析
│   ├── run_exploration.py           #   阶段入口
│   ├── dataset_overview.py          #   2.1 数据集概览（18.8万×13维）
│   ├── target_analysis.py           #   2.2 目标变量分析（log1p变换）
│   ├── correlation_heatmap.py       #   2.3 皮尔逊相关系数热力图
│   ├── feature_transform.py         #   2.4 4种变换策略对比（2×4面板图）
│   └── feature_validation.py        #   2.5 衍生特征四维验证
│       (Permutation + Bootstrap + PDP + Ablation)
│
├── preprocessing/                   # 数据预处理与特征工程
│   ├── run_preprocessing.py         #   阶段入口（生成特征矩阵 + 序列化Pipeline）
│   ├── data_cleaning.py             #   3.1 数据清洗（品牌标准化/IQR截尾/MAR填充）
│   ├── encoding_comparison.py       #   3.2 编码对比（Drop vs OHE vs Target Encoding）
│   ├── feature_engineering.py       #   3.3 特征工程（4衍生+OOF目标编码+Z-score）
│   └── feature_selection.py         #   3.4 VIF+MI两阶段特征筛选（21→11维）
│
├── modeling/                        # 模型构建与评估
│   ├── run_modeling.py              #   阶段入口（训练+对比+保存最佳模型）
│   ├── model_training.py            #   4.1-4.2 模型选择与训练
│   ├── hyperparameter_tuning.py     #   4.2 超参数调优（手动+Optuna贝叶斯）
│   ├── model_comparison.py          #   4.3 多模型性能对比（雷达图+表格）
│   └── model_explainability.py      #   4.4 SHAP可解释性 + 高级审计
│
├── analysis/                        # 结果分析与决策优化
│   ├── run_analysis.py              #   阶段入口
│   ├── prediction_analysis.py       #   5.1 预测vs实际图 + 价格区间分析 + 品牌残差
│   └── business_insights.py         #   5.2 量化业务建议（3条核心建议）
│
├── application/                     # 系统设计与实现
│   ├── backend/                     #   FastAPI 后端
│   │   ├── main.py                  #     API入口 (CORS + lifespan)
│   │   ├── schemas.py               #     Pydantic 请求/响应模型
│   │   ├── pipeline_loader.py       #     单例加载 Pipeline 对象
│   │   ├── prediction_service.py    #     推理服务（校验→变换→预测→SHAP）
│   │   └── requirements.txt         #     后端依赖
│   ├── frontend/                    #   Vue 3 + Element Plus 前端
│   │   ├── package.json
│   │   ├── vite.config.js           #     Vite配置 + API代理
│   │   ├── index.html
│   │   └── src/
│   │       ├── main.js              #     Vue入口（Element Plus注册）
│   │       ├── App.vue              #     根组件（左右双栏布局）
│   │       ├── api/index.js         #     axios API封装
│   │       ├── components/
│   │       │   ├── InputForm.vue    #     输入表单（品牌→车型联动+9字段）
│   │       │   ├── ResultCard.vue   #     结果卡片（预测价+市场对比+分位数）
│   │       │   └── ShapChart.vue    #     SHAP贡献水平条形图
│   │       └── styles/main.css      #     全局样式
│   ├── run_backend.py               #   启动后端 (uvicorn :8000)
│   └── run_frontend.py              #   启动前端 (Vite :5173)
│
│
├── src/                             # 核心模块库
│   ├── preprocess.py                #   AdvancedUsedCarPreprocessor
│   ├── features.py                  #   HighScoreFeatureEngineer（4衍生+OOF编码+Z-score）
│   ├── selection.py                 #   FeatureSelector（VIF>10淘汰 + MI<5%均值淘汰）
│   ├── models.py                    #   UsedCarModelFactory（5模型+Stacking集成）
│   ├── train.py                     #   ModelTrainer（分层CV+雷达图+参数曲线）
│   ├── tune_all_models.py           #   超参数优化（手动网格 + Optuna TPE贝叶斯）
│   ├── evaluate.py                  #   高级审计（Top-10极端误差 + 价格/品牌公平性）
│   └── explainability.py            #   SHAP可解释性（Summary/Dependence/Waterfall）
│
├── data/
│   ├── raw/                         # train.csv + test.csv（原始188,533+125,690条）
│   └── processed/                   # train_features.csv + test_features.csv（11维特征矩阵）
│
├── models/                          # 训练产物
│   ├── lightgbm_model.pkl           #   最佳模型（R²=0.6622, MAE=$17,189）
│   ├── xgboost_model.pkl
│   ├── catboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── ridge_model.pkl
│   ├── stacking_model.pkl
│   ├── preprocessor.pkl             #   已拟合的预处理管道
│   ├── feature_engineer.pkl         #   已拟合的特征工程管道（含brand/model映射+scaler）
│   ├── feature_selector.pkl         #   已拟合的特征选择器
│   ├── feature_names.json           #   最终11个特征列名
│   ├── brand_list.json              #   57个可用品牌
│   ├── model_list.json              #   1,751个车型（按品牌分组）
│   ├── model_comparison.csv         #   5模型5-fold CV指标对比
│   ├── feature_importance_summary.csv
│   ├── optimized_params.json        #   各模型最优超参数
│   └── stacking_result.csv
│
└── reports/
    ├── figures/                     # 26张高质量可视化图表（PNG, 300dpi）
    ├── shap_feature_importance.csv  # SHAP特征重要性
    ├── top10_extreme_errors.xlsx    # Top-10极端误差样本
    ├── brand_retention_ranking.csv  # 品牌保值率排名
    ├── price_segment_strategy.csv   # 分层定价策略表
    └── 项目报告（模版）.docx
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- uv（Python 包管理）

### 安装与数据准备

```bash
# 1. 安装 Python 依赖
uv sync

# 2. 将 train.csv 和 test.csv 放入 data/raw/

# 3. 一键运行全流程
python run_all.py
```

### 启动 Web 系统

```bash
# 终端1: 启动 FastAPI 后端
python application/run_backend.py
# → http://localhost:8000
# → API文档: http://localhost:8000/docs

# 终端2: 启动 Vue 前端
python application/run_frontend.py
# → http://localhost:5173
```

### 分步运行

```bash
python exploration/run_exploration.py       # 数据探索
python preprocessing/run_preprocessing.py   # 预处理+特征工程
python modeling/run_modeling.py             # 模型训练+对比
python analysis/run_analysis.py             # 结果分析+业务建议
```

---

## API 接口文档

启动后端后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/brands` | 获取57个品牌列表 |
| `GET` | `/api/models?brand=Toyota` | 按品牌筛选车型 |
| `GET` | `/api/stats` | 市场统计（均价/样本量/模型指标） |
| `POST` | `/api/predict` | **核心接口**：预测价格 + SHAP解释 |
| `POST` | `/api/validate` | 仅校验输入参数（不预测） |

### POST /api/predict 示例

请求：
```json
{
  "brand": "Toyota",
  "model": "Camry",
  "model_year": 2020,
  "milage": 45000,
  "engine_hp": 203,
  "engine_liter": 2.5,
  "fuel_type": "Gasoline",
  "accident": "None reported",
  "clean_title": "Yes"
}
```

响应：
```json
{
  "predicted_price": 26705,
  "price_tier": "中端实用车",
  "market_avg_price": 43878,
  "percentile": 42,
  "shap_explanation": [
    { "feature": "milage", "display_name": "行驶里程", "contribution": -4299, "direction": "negative" },
    { "feature": "model_encoded", "display_name": "车型保值率", "contribution": -2523, "direction": "negative" },
    { "feature": "brand_encoded", "display_name": "品牌溢价", "contribution": -2023, "direction": "negative" },
    { "feature": "accident_status_No_Accident", "display_name": "无事故记录", "contribution": -1592, "direction": "negative" },
    { "feature": "annual_milage", "display_name": "年均行驶里程", "contribution": 1343, "direction": "positive" }
  ],
  "natural_language": [
    "[-] 行驶里程：拉低 -$4,299，是价格下降的主要因素",
    "[-] 车型保值率：拉低 -$2,523",
    "[-] 品牌溢价：拉低 -$2,023",
    "[-] 无事故记录：拉低 -$1,592",
    "[+] 年均行驶里程：贡献 +$1,343，对价格有正向支撑"
  ],
  "validation_warnings": []
}
```

---

## 核心亮点

### 数据预处理

- **品牌名称标准化**：Mercedes→Mercedes-Benz, Land→Land Rover 等，稀有品牌（<15样本）归拢为 "Other"
- **品牌级局部 IQR 截尾**：按品牌分组学习边界，避免豪车被全局阈值误杀
- **MAR 缺失值填补**：Tesla + Electric引擎 → 电动，事故缺失 → "Unknown"独立类别
- **3种编码方式定量对比**：Drop vs One-Hot vs Target Encoding OOF，最终选择 OOF Target Encoding

### 特征工程

- **4个经验证有效的衍生特征**：

| 特征 | 公式 | SHAP贡献 | 业务含义 |
|------|------|----------|---------|
| `hp_per_year` | engine_hp / (car_age + 1) | 0.253 (#1) | 发动机老化速度 |
| `annual_milage` | milage / (car_age + 1) | 0.031 (#9) | 年均磨损强度 |
| `power_density` | engine_hp / engine_liter | 0.050 (#5) | 升功率·技术溢价 |
| `car_age_squared` | car_age² | 0.068 (#3) | 非线性折旧加速 |

- **VIF + MI 两阶段特征筛选**：21维 → 11维（降维 47.6%）
- **OOF嵌套交叉验证目标编码**：消除标签泄露 + 拉普拉斯平滑防过拟合

### 模型训练

| 模型 | R² | MAE | RMSE | MAPE | 训练时间 |
|------|------|------|------|------|---------|
| **XGBoost** | **0.6626** | **$17,170** | $73,209 | 37.0% | 72s |
| LightGBM | 0.6622 | $17,189 | $73,220 | 37.1% | 127s |
| CatBoost | 0.6593 | $17,261 | $73,361 | 37.3% | 48s |
| Random Forest | 0.6558 | $17,311 | $73,299 | 37.6% | 358s |
| Ridge | 0.6189 | $18,189 | $74,300 | 40.2% | 0.2s |
| Stacking | 0.6408 | $17,630 | — | — | — |

- **5折分层交叉验证**（按价格十分位数分层）
- **Optuna TPE贝叶斯优化** + 手动参数曲线（ridge α / lightgbm num_leaves + learning_rate / elasticnet 热力图）
- **多维度雷达图**：R² + MAE⁻¹ + RMSE⁻¹ + MAPE⁻¹ + 训练时间⁻¹

### 模型评估

- **Top-10极端误差样本分析**：导出 Excel，分析共性（高估/低估方向、价格区间分布）
- **价格区间公平性审计**：低端/中端/高档/豪华四段 MAE + MAPE + 系统偏差
- **品牌公平性审计**：Top-5主流品牌的残差分析
- **SHAP 可解释性**：Summary Plot + Importance Bar + Dependence Plot + Waterfall 单样本分解

### 业务建议（

3条量化核心建议（详见 `analysis/business_insights.py`）：

1. **定价核心驱动**：hp_per_year（发动机状态）是第一要素，里程次之。低里程区每千英里贬值 $334，高里程区仅 $25
2. **品牌保值率排名**：Bentley/Porsche/Lamborghini 被系统性低估20-32% → 收购机会
3. **分层定价策略**：$3.5-7.5万区间 MAPE=27% → 模型可直接定价；<$1.5万 MAPE=63.6% → 需人工判断

### 系统设计

- **FastAPI 后端**：加载训练时序列化的 Pipeline 对象（preprocessor + feature engineer + selector），保证推理与训练完全一致
- **Vue 3 + Element Plus 前端**：品牌→车型联动下拉框、表单实时校验、SHAP 贡献可视化
- **前后端分离**：RESTful API，Vite 开发代理，支持独立部署

---

## 特征清单（最终11维）

| # | 特征 | 类型 | SHAP | 说明 |
|---|------|------|------|------|
| 1 | hp_per_year | 衍生 | 0.253 | 年均马力保有量（#1关键特征） |
| 2 | milage | 基础 | 0.167 | 行驶里程（Z-score） |
| 3 | car_age_squared | 衍生 | 0.068 | 车龄平方（非线性折旧） |
| 4 | brand_encoded | 编码 | 0.053 | 品牌OOF目标编码 |
| 5 | power_density | 衍生 | 0.050 | 升功率（HP/L） |
| 6 | engine_hp | 基础 | 0.049 | 发动机马力（Z-score） |
| 7 | model_encoded | 编码 | 0.038 | 车型OOF目标编码 |
| 8 | annual_milage | 衍生 | 0.031 | 年均行驶里程 |
| 9 | accident_status_No_Accident | OHE | 0.014 | 无事故记录（二值） |
| 10 | fuel_type_Electric | OHE | 0.002 | 电动车型（二值） |
| 11 | is_clean_title | 二值 | <0.001 | 清晰产权 |

> **已剔除特征**（VIF>10或MI<5%均值）：car_age, engine_liter, brand_x_milage, fuel_type_Hybrid, fuel_type_Gasoline, fuel_type_E85 Flex Fuel, fuel_type_Unknown, fuel_type_Plug-In Hybrid, accident_status_Unknown

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn, LightGBM, XGBoost, CatBoost |
| 超参数优化 | Optuna（TPE 贝叶斯优化） |
| 可解释性 | SHAP（TreeExplainer） |
| 可视化 | Matplotlib, Seaborn |
| 后端 | FastAPI + Uvicorn |
| 前端 | Vue 3 + Vite + Element Plus + axios |
| 包管理 | uv（Python）+ npm（前端） |
