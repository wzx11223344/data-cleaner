# data-cleaner
[![CI](https://github.com/wzx11223344/data-cleaner/actions/workflows/ci.yml/badge.svg)](https://github.com/wzx11223344/data-cleaner/actions/workflows/ci.yml)


数据清洗工具箱 - 10个高级算法驱动的数据预处理工具集（纯Python标准库实现）

## 功能概览

| # | 函数名 | 算法原理 | 复杂度 |
|---|--------|----------|--------|
| 1 | `data_profiling` | 类型推断 + 统计量计算 + 四分位/偏度/峰度 | O(n*m) |
| 2 | `outlier_detector_zscore` | 修改版Z-score（基于MAD中位数绝对偏差） | O(n log n)排序 |
| 3 | `outlier_detector_isolation_forest` | Isolation Forest（随机选特征+分割点→路径长度→异常得分） | O(n*t*log(n)) |
| 4 | `missing_value_imputer` | KNN填充(距离加权)/均值/中位数/众数/回归填充 | KNN: O(n^2*k) |
| 5 | `duplicate_detector` | 精确匹配(哈希) + 模糊匹配(编辑距离+数值容差) + 并查集 | 精确O(n), 模糊O(n^2) |
| 6 | `data_normalizer` | Min-Max/Z-Score/Robust(IQR)/Decimal/Log五种标准化 | O(n*m) |
| 7 | `correlation_analyzer` | Pearson/Spearman/Kendall三种相关系数 + p值显著性检验 | O(n^2*m^2) |
| 8 | `data_type_converter` | 智能类型推断(int/float/bool/datetime/category) + 格式处理 | O(n*m) |
| 9 | `data_quality_scorer` | 5维度评分(完整性/唯一性/一致性/准确性/时效性) | O(n*m) |
| 10 | `data_export_pipeline` | CSV/JSON/TSV/Excel XML格式 + GZIP压缩 | O(n*m) |

## 算法详解

### 1. 数据画像分析 (`data_profiling`)
- **类型推断**: 尝试将列值转换为int/float/bool/datetime，确定最佳类型
- **统计量**: 唯一值数、缺失率、最小/最大/均值/标准差
- **四分位数**: 使用线性插值法计算 Q1/Q2/Q3
- **偏度(Pearson)**: `g1 = (m3 / m2^1.5)`，m2=二阶中心矩，m3=三阶中心矩
- **峰度(超额)**: `g2 = (m4 / m2^2) - 3`
- **高频值**: Counter统计Top-10
- **复杂度**: O(n*m)，n=行数，m=列数

### 2. Z-score异常检测 (`outlier_detector_zscore`)
- **修改版Z-score**: 基于MAD（中位数绝对偏差）而非标准差
  - `MAD = median(|x_i - median(x)|)`
  - `z_i = 0.6745 * (x_i - median) / MAD`
  - MAD=0时回退到标准Z-score
- **异常标记**: |z_i| > threshold (默认3)
- **报告**: 异常值索引、值、Z-score、异常比例
- **复杂度**: 排序O(n log n)

### 3. Isolation Forest异常检测 (`outlier_detector_isolation_forest`)
- **算法原理**: 异常点"稀疏且少数"，更容易被孤立（路径更短）
- **构建流程**:
  1. 从数据中随机采样 subset (size=sample_size)
  2. 随机选择一个特征
  3. 在该特征的最小/最大值之间随机选择分割点
  4. 递归分割直到：数据点被孤立 / 达到最大深度(log2(sample_size))
  5. 重复 n_trees 次，构建森林
- **路径长度**: 每个数据点在每棵树中的路径长度（包括外部路径+内部路径修正）
- **异常得分**: `s = 2^(-E(h) / c(n))`
  - `E(h)`: 平均路径长度
  - `c(n) = 2 * H(n-1) - 2*(n-1)/n`: 二叉搜索树的平均路径长度
  - `H(i) = ln(i) + 0.5772156649` (欧拉常数)
  - s > 0.5 倾向异常，s → 1 强异常
- **复杂度**: O(n*t*log(n))，t=树数

### 4. 缺失值填充 (`missing_value_imputer`)
- **KNN填充**:
  - 计算含缺失值的记录与所有完整记录的欧氏距离
  - 选K个最近邻
  - 距离倒数加权: `w_i = 1 / (d_i + epsilon)`
  - 加权平均填充
- **均值/中位数/众数填充**: 基于列统计量
- **回归填充**:
  - 以缺失列作为目标变量Y
  - 以其他数值列作为特征X
  - 最小二乘法求解: `β = (X^T X)^{-1} X^T Y`（使用高斯消元法解方程组）
  - 预测缺失值
- **复杂度**: KNN O(n^2*k)，回归 O(n^3)高斯消元

### 5. 数据去重 (`duplicate_detector`)
- **精确匹配**: 对每条记录计算哈希值，哈希相同即为重复
- **模糊匹配**:
  - 字符串字段: Levenshtein编辑距离 `_levenshtein(a, b)`，归一化为相似度
  - 数值字段: 相对容差 `|a-b| / max(|a|,|b|) < tolerance`
  - 综合相似度: 各字段相似度的加权平均
- **并查集(Union-Find)**: 将相似记录合并为同一组
- **复杂度**: 精确O(n)，模糊O(n^2*字段数)

### 6. 数据标准化 (`data_normalizer`)
- **Min-Max**: `x' = (x - min) / (max - min)`
- **Z-Score**: `x' = (x - mean) / std`
- **Robust Scaling**: `x' = (x - median) / IQR` (基于四分位距，抗异常值)
- **Decimal Scaling**: `x' = x / 10^j`，j为使|max(x')| < 1的最小整数
- **Log Scaling**: `x' = log(x + 1)` 或 `x' = sign(x) * log(|x| + 1)`
- **复杂度**: O(n*m)

### 7. 相关性分析 (`correlation_analyzer`)
- **Pearson相关系数**: 
  - `r = Σ(xi-x̄)(yi-ȳ) / sqrt(Σ(xi-x̄)^2 * Σ(yi-ȳ)^2)`
  - 衡量线性相关性
- **Spearman秩相关系数**:
  - 将值转换为秩，计算Pearson相关
  - `ρ = 1 - 6Σd_i^2 / (n(n^2-1))`，d_i=秩差
  - 衡量单调相关性
- **Kendall Tau**:
  - 统计一致对(C)和不一致对(D)
  - `τ = (C - D) / (n*(n-1)/2)`
  - 衡量序数相关性
- **p值显著性检验**:
  - 使用标准正态CDF近似（Abramowitz & Stegun公式7.1.26）
  - `z = r * sqrt(n-2) / sqrt(1-r^2)`，双侧p值
- **复杂度**: O(n^2*m^2)，n=行数，m=列数

### 8. 智能类型转换 (`data_type_converter`)
- **类型推断**: 尝试 int → float → bool → datetime → category 的优先级
- **格式处理**:
  - int: 去除千分位逗号、空格
  - float: 处理科学计数法
  - bool: 识别 "true/false/yes/no/1/0"
  - datetime: 尝试多种日期格式 (%Y-%m-%d, %Y/%m/%d, %d-%m-%Y等)
- **批量转换**: 对整列统一应用推断的最佳类型
- **复杂度**: O(n*m)

### 9. 数据质量评分 (`data_quality_scorer`)
- **5个评分维度**:
  1. **完整性** (30%): 非空率 = 1 - 缺失率
  2. **唯一性** (20%): 唯一记录率 = 唯一行数 / 总行数
  3. **一致性** (20%): 数据类型一致性 = 一致类型列数 / 总列数
  4. **准确性** (20%): 非异常率 = 1 - 异常值比例
  5. **时效性** (10%): 最近更新时间评分
- **综合评分**: 加权平均，0-100分
- **等级**: A(90+), B(80+), C(70+), D(60+), F(<60)
- **报告**: 各维度得分 + 问题描述 + 改进建议
- **复杂度**: O(n*m)

### 10. 数据导出管线 (`data_export_pipeline`)
- **支持格式**:
  - CSV: 逗号分隔
  - JSON: 结构化JSON数组
  - TSV: 制表符分隔
  - JSONL: 每行一个JSON对象
  - Excel XML: Excel 2003 XML格式(SpreadsheetML)
- **压缩**: GZIP压缩（使用gzip模块）
- **多格式同时导出**: 一次调用导出多种格式
- **复杂度**: O(n*m)

## 安装

无需安装额外依赖，仅使用Python标准库（math, csv, json, re, gzip, random, statistics, collections, datetime等）。

**注意: 不使用numpy/pandas，所有算法从零实现。**

## 使用示例

```python
from main import data_profiling, outlier_detector_isolation_forest, correlation_analyzer

# 数据画像
data = [{"name": "张三", "age": 25, "salary": 8000}, {"name": "李四", "age": 30, "salary": 12000}]
profile = data_profiling(data)
print(profile["column_stats"]["age"]["mean"])  # 27.5

# Isolation Forest异常检测
values = [[1.0], [2.0], [3.0], [100.0]]  # 100.0是异常
result = outlier_detector_isolation_forest(values, n_trees=50)
print(result["anomaly_scores"])  # 100.0的得分最高

# 相关性分析
corr = correlation_analyzer(data, method="pearson")
print(corr["matrix"])  # 相关系数矩阵
```

## License

MIT
