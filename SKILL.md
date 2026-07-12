---
name: data-cleaner-zx
displayName: 数据清洗工具箱
version: 1.0.1
summary: 10个数据工具：数据画像/异常检测/缺失处理/去重/标准化/编码/质量报告/相关性/类型转换/导出
tags: [data, cleaning, analysis, pandas]
license: MIT
---

# 数据清洗工具箱 (data-cleaner)

## 简介

data-cleaner 是一套包含10个数据清洗与分析工具的 Python 技能包，覆盖数据预处理全流程，包括数据画像、异常值检测、缺失值处理、去重、标准化、编码、质量报告等。

## 功能列表

| # | 函数名 | 功能描述 |
|---|--------|----------|
| 1 | `load_and_profile` | 数据加载与画像（行数/列数/类型/缺失率） |
| 2 | `detect_outliers` | 异常值检测（IQR/Z-score） |
| 3 | `handle_missing` | 缺失值处理（删除/均值/中位数/插值） |
| 4 | `remove_duplicates` | 去重 |
| 5 | `normalize_data` | 标准化（Min-Max/Z-score/Robust） |
| 6 | `encode_categorical` | 分类编码（Label/OneHot/Target） |
| 7 | `generate_profile_report` | 数据质量报告（HTML格式） |
| 8 | `correlation_analysis` | 相关性分析 |
| 9 | `data_type_converter` | 类型转换 |
| 10 | `export_cleaned_data` | 导出清洗后数据 |

## 安装

```bash
pip install pandas numpy scikit-learn
```

## 使用示例

```python
from main import load_and_profile, detect_outliers, handle_missing

# 加载并画像
result = load_and_profile("data.csv")
print(result["profile"]["shape"])

# 检测异常值
outliers = detect_outliers(result["data"], method="iqr")

# 处理缺失值
cleaned = handle_missing(result["data"], strategy="mean")
print(f"填充了 {cleaned['filled_count']} 个缺失值")
```

## 依赖

- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: 标准化与编码工具

## License

MIT
