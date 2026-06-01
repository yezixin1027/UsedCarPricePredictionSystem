# 🚗 Used Car Price Prediction System · 二手车价格预测系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.3+-yellow.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于真实二手车交易数据的**工业级价格预测系统**，涵盖从原始脏数据清洗、高维特征工程到建模特征矩阵持久化的完整机器学习 Pipeline。

---

## 📋 目录

- [项目背景](#项目背景)
- [核心亮点](#核心亮点)
- [项目结构](#项目结构)
- [环境与依赖](#环境与依赖)
- [快速开始](#快速开始)
- [Pipeline 详解](#pipeline-详解)
  - [阶段 1：数据加载](#阶段-1数据加载)
  - [阶段 2：高级预处理](#阶段-2高级预处理)
  - [阶段 3：特征工程](#阶段-3特征工程)
  - [阶段 4：持久化输出](#阶段-4持久化输出)
- [EDA 可视化](#eda-可视化)
- [数据流图](#数据流图)
- [后续扩展](#后续扩展)

---

## 项目背景

二手车市场存在严重的信息不对称问题，买卖双方对车辆真实价值的判断往往依赖主观经验。本项目基于真实二手车交易数据集，构建了一套**端到端的特征工程 Pipeline**，通过对原始异构数据的系统化清洗与特征提取，为后续机器学习建模提供高质量、无数据泄露的特征矩阵。

### 技术栈

| 层级 | 技术选型 |
|------|----------|
| 语言 | Python 3.10+ |
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn（TransformerMixin 管道化封装） |
| 可视化 | Matplotlib, Seaborn |
| 包管理 | uv（pyproject.toml） |

---

## 核心亮点

### 🛡️ 防数据泄露设计
- **五折嵌套隔离目标编码（Out-of-Fold Target Encoding）**：对品牌（brand）等高基数分类变量做目标编码时，严格采用 OOF 交叉验证方式计算编码值，杜绝训练集信息向验证集泄露。
- **fit / transform 分离**：所有预处理器和特征工程器均继承 `sklearn.base.BaseEstimator` 与 `TransformerMixin`，确保统计量仅在训练集上学习，测试集仅做静态映射。

### 🧹 自适应数据清洗
- **品牌名称标准化**：自动纠正拼写歧义（如 `Mercedes → Mercedes-Benz`、`VW → Volkswagen`）。
- **正则引擎特征抽取**：从非结构化引擎文本中自适应提取马力（HP）与排量（L），支持缺失 "L" 标注的备用匹配模式。
- **MAR 机制缺失值填充**：结合业务逻辑联动填充（如 Tesla 品牌自动归为 Electric 燃料类型）。
- **品牌级局部 IQR 截尾**：按品牌分组计算里程的局部 IQR 边界，避免高性能豪车被全局阈值"误杀"。

### 🔧 创新衍生特征
- **年均行驶里程** `annual_milage = milage / (car_age + 1)`：融合车龄与里程的复合特征，更精准反映车辆使用强度。
- **事故记录提炼**：将缺失值独立编码为 `Unknown` 类别，保留潜在欺诈行为模式供算法学习。

---

## 项目结构

```
UsedCarPricePredictionSystem/
├── main.py                          # 程序入口
├── run_pipeline.py                  # 🔥 主 Pipeline 执行脚本
├── pyproject.toml                   # 项目配置与依赖声明
├── uv.lock                          # 依赖锁定文件
├── .gitignore
├── .python-version                  # Python 版本锁定
│
├── src/                             # 核心源码模块
│   ├── config.py                    # 全局路径 & 参数配置
│   ├── preprocess.py                # 高级数据预处理器
│   └── features.py                  # 高评分特征工程器
│
├── eda_plots/                       # 探索性数据分析脚本
│   ├── eda_heatmap.py               # 特征相关性热力图
│   └── eda_target.py                # 目标变量分布对比
│
├── data/
│   ├── raw/                         # 原始数据
│   │   ├── train.csv                # 训练集
│   │   └── test.csv                 # 测试集
│   └── processed/                   # 清洗后的特征矩阵
│       ├── train_features.csv
│       └── test_features.csv
│
├── models/                          # 模型持久化目录
│
└── reports/
    └── figures/                     # 可视化输出
        ├── feature_correlation_heatmap.png
        └── price_distribution_comparison.png
```

---

## 环境与依赖

### 前置要求

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv)（推荐，用于快速依赖管理）

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/yezixin1027/UsedCarPricePredictionSystem.git
cd UsedCarPricePredictionSystem

# 2. 创建虚拟环境并安装依赖（使用 uv）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `pandas` | >= 2.3.3 | 数据加载与 DataFrame 操作 |
| `numpy` | >= 2.2.6 | 数值计算 |
| `scikit-learn` | >= 1.7.2 | Pipeline 基类封装、标准化、交叉验证 |
| `matplotlib` | >= 3.10.9 | 基础绑图 |
| `seaborn` | >= 0.13.2 | 统计可视化（热力图、分布图） |
| `notebook` | >= 7.5.6 | Jupyter Notebook 支持 |

---

## 快速开始

### 运行完整 Pipeline

```bash
python run_pipeline.py
```

执行后将依次完成：
1. 加载原始 CSV 数据
2. 运行 `AdvancedUsedCarPreprocessor` 清洗预处理
3. 运行 `HighScoreFeatureEngineer` 特征工程
4. 将最终特征矩阵持久化至 `data/processed/`

### 生成 EDA 图表

```bash
# 目标变量分布对比图（原始 vs Log 变换）
python eda_plots/eda_target.py

# 特征相关性热力图
python eda_plots/eda_heatmap.py
```

生成的图表自动保存至 `reports/figures/` 目录。

---

## Pipeline 详解

### 阶段 1：数据加载

```python
train_df = pd.read_csv(TRAIN_PATH)   # 训练集
test_df  = pd.read_csv(TEST_PATH)    # 测试集

# 严格拆分自变量与响应变量
X_train_raw = train_df.drop(columns=['price'])
y_train_raw = train_df['price']
```

### 阶段 2：高级预处理

`AdvancedUsedCarPreprocessor` 执行以下清洗步骤：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 品牌标准化 | 统一非标准拼写（`Land → Land Rover`、`Chevy → Chevrolet`） |
| 2 | 稀有品牌归拢 | 样本量 < 15 的品牌合并为 `Other` |
| 3 | 引擎特征抽取 | 正则提取 `engine_hp`（马力）、`engine_liter`（排量） |
| 4 | 品牌级 IQR 截尾 | 按品牌局部截断 `milage` 极端异常值（IQR × 3.0） |
| 5 | 业务联动填充 | Tesla → Electric；缺失燃料类型归为 `Unknown` |
| 6 | 事故记录编码 | `accident` → `accident_status`（Has_Accident / No_Accident / Unknown） |
| 7 | 车龄计算 | `car_age = 2026 - model_year` |
| 8 | 特征裁剪 | 剔除高基数/冗余列（`engine`、`model`、`transmission` 等） |

### 阶段 3：特征工程

`HighScoreFeatureEngineer` 执行以下变换：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 年均里程衍生 | `annual_milage = milage / (car_age + 1)` |
| 2 | OOF 目标编码 | 5 折交叉验证隔离编码 `brand` → `brand_encoded` |
| 3 | One-Hot 编码 | 低基数分类变量（`fuel_type`、`accident_status`） |
| 4 | Z-score 标准化 | 连续特征零均值单位方差变换 |

### 阶段 4：持久化输出

```python
train_final_output['log_price'] = np.log1p(y_train_raw)
train_final_output.to_csv("data/processed/train_features.csv", index=False)
X_test_final.to_csv("data/processed/test_features.csv", index=False)
```

输出特征矩阵可直接用于 XGBoost、LightGBM、Random Forest 等任意监督学习模型的训练与预测。

---

## EDA 可视化

### 目标变量分布（Log 变换前后对比）

![price_distribution](reports/figures/price_distribution_comparison.png)

原始价格呈严重右偏分布，经过 `log1p` 变换后接近正态分布，更适合线性模型与梯度下降优化。

### 特征相关性热力图

![correlation_heatmap](reports/figures/feature_correlation_heatmap.png)

展示核心连续特征（行驶里程、发动机马力、排量、车龄）与对数价格之间的皮尔逊相关系数。

---

## 数据流图

```
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│  data/raw/   │ ──► │ AdvancedUsedCarPreprocessor │ ──► │ HighScoreFeatureEngineer │
│ train.csv    │     │  · 品牌清洗               │     │  · 年均里程衍生           │
│ test.csv     │     │  · 引擎特征抽取           │     │  · OOF 目标编码           │
└──────────────┘     │  · IQR 局部截尾           │     │  · One-Hot 编码           │
                     │  · 缺失值填充             │     │  · Z-score 标准化         │
                     │  · 车龄计算               │     └──────────┬───────────────┘
                     └─────────────────────────┘                │
                                                                ▼
                                                     ┌────────────────────┐
                                                     │  data/processed/   │
                                                     │  train_features.csv│
                                                     │  test_features.csv  │
                                                     └────────────────────┘
```

---

## 后续扩展

- [ ] 集成 XGBoost / LightGBM 模型训练模块
- [ ] 添加超参数网格搜索与交叉验证
- [ ] 构建模型可解释性分析（SHAP / Permutation Importance）
- [ ] 开发 Flask / FastAPI 推理服务接口
- [ ] 添加 CI/CD 自动化测试与部署

---

## 作者

- **Ye** — [@yezixin1027](https://github.com/yezixin1027)

---

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件（若有）。

---

<p align="center">
  <sub>Built with ❤️ for Data Science</sub>
</p>
