# data-cleaner

数据清洗工具箱 - 10个数据清洗与分析工具集

## 功能概览

- **数据加载与画像** - 自动分析数据形状、类型、缺失率、统计信息
- **异常值检测** - IQR和Z-score两种检测方法
- **缺失值处理** - 删除/均值/中位数/众数/插值/前向/后向填充
- **去重** - 支持全列或指定列去重
- **数据标准化** - Min-Max/Z-score/Robust三种方法
- **分类编码** - Label/OneHot/Target编码
- **数据质量报告** - 生成HTML格式可视化报告
- **相关性分析** - Pearson/Spearman/Kendall三种方法
- **类型转换** - int/float/str/bool/datetime互转
- **数据导出** - CSV/Excel/JSON/Parquet四种格式

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from main import load_and_profile, generate_profile_report

# 加载数据
result = load_and_profile("data.csv")

# 生成质量报告
report = generate_profile_report(result["data"])
with open("quality_report.html", "w") as f:
    f.write(report["report"])
```

## License

MIT
