---
name: data-cleaner-zx
displayName: 数据清洗工具箱
version: 2.0.0
summary: 10个高级算法数据工具：数据画像(偏度峰度)/MAD异常检测/Isolation Forest/KNN+回归填充/模糊去重(并查集)/5种标准化/Pearson+Spearman+Kendall相关性/智能类型转换/5维度质量评分/多格式导出
tags: [data, cleaning, isolation-forest, knn, correlation, data-quality, statistics]
license: MIT
---

# 数据清洗工具箱 (data-cleaner)

## 简介

data-cleaner 是一套包含10个高级算法驱动的数据清洗与分析工具的 Python 技能包。**不使用numpy/pandas，所有算法从零实现。** 涵盖数据画像、异常值检测(Isolation Forest)、缺失值填充(KNN+回归)、模糊去重(并查集)、五种标准化、三种相关系数(Pearson/Spearman/Kendall)、智能类型转换、五维度质量评分、多格式导出等场景。

## 功能列表

| # | 函数名 | 算法原理 | 复杂度 |
|---|--------|----------|--------|
| 1 | `data_profiling` | 类型推断 + 统计量 + 四分位/偏度/峰度 | O(n*m) |
| 2 | `outlier_detector_zscore` | 修改版Z-score（MAD中位数绝对偏差） | O(n log n) |
| 3 | `outlier_detector_isolation_forest` | Isolation Forest（随机分割→路径长度→异常得分） | O(n*t*log(n)) |
| 4 | `missing_value_imputer` | KNN(距离加权)/均值/中位数/众数/回归(高斯消元) | KNN: O(n^2*k) |
| 5 | `duplicate_detector` | 精确匹配(哈希) + 模糊匹配(编辑距离) + 并查集 | 精确O(n), 模糊O(n^2) |
| 6 | `data_normalizer` | Min-Max/Z-Score/Robust(IQR)/Decimal/Log | O(n*m) |
| 7 | `correlation_analyzer` | Pearson/Spearman/Kendall + p值显著性检验 | O(n^2*m^2) |
| 8 | `data_type_converter` | 智能类型推断(int/float/bool/datetime/category) | O(n*m) |
| 9 | `data_quality_scorer` | 5维度评分(完整性/唯一性/一致性/准确性/时效性) | O(n*m) |
| 10 | `data_export_pipeline` | CSV/JSON/TSV/Excel XML + GZIP压缩 | O(n*m) |

## 安装

无需安装额外依赖，仅使用Python标准库（math, csv, json, re, gzip, random, statistics, collections, datetime等）。

**注意: 不使用numpy/pandas，所有算法从零实现。**

## 使用示例

```python
from main import data_profiling, outlier_detector_isolation_forest, correlation_analyzer

# 数据画像
data = [{"name": "张三", "age": 25, "salary": 8000}, {"name": "李四", "age": 30, "salary": 12000}]
profile = data_profiling(data)
print(profile["column_stats"]["age"]["mean"])

# Isolation Forest异常检测
values = [[1.0], [2.0], [3.0], [100.0]]
result = outlier_detector_isolation_forest(values, n_trees=50)
print(result["anomaly_scores"])

# 相关性分析
corr = correlation_analyzer(data, method="pearson")
print(corr["matrix"])
```

## 依赖

无外部依赖（仅使用Python标准库，不使用numpy/pandas）

## License

MIT
