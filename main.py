"""
数据清洗工具箱 (data-cleaner)

提供10个数据清洗与分析工具，包括数据加载与画像、异常值检测、缺失值处理、
去重、数据标准化、分类编码、数据质量报告、相关性分析、类型转换和数据导出。
基于 pandas/numpy/scikit-learn 构建，适用于数据预处理全流程。

主要功能:
    - load_and_profile: 数据加载与画像
    - detect_outliers: 异常值检测（IQR/Z-score）
    - handle_missing: 缺失值处理
    - remove_duplicates: 去重
    - normalize_data: 标准化（Min-Max/Z-score/Robust）
    - encode_categorical: 分类编码（Label/OneHot/Target）
    - generate_profile_report: 数据质量报告（HTML格式）
    - correlation_analysis: 相关性分析
    - data_type_converter: 类型转换
    - export_cleaned_data: 导出清洗后数据

依赖:
    - pandas: 数据处理
    - numpy: 数值计算
    - scikit-learn: 机器学习工具（标准化/编码）
"""

import json
import os
from datetime import datetime


# =============================================================================
# 1. 数据加载与画像
# =============================================================================
def load_and_profile(csv_path):
    """
    加载CSV文件并生成数据画像。

    数据画像包括行数、列数、数据类型、缺失率、基本统计信息等。

    Args:
        csv_path (str): CSV文件路径。

    Returns:
        dict: 包含以下键的字典:
            - "data": pandas DataFrame（内存中）
            - "profile": 画像字典:
                - "shape": {"rows": 行数, "columns": 列数}
                - "columns": 列信息列表，每项为:
                    {"name": 列名, "dtype": 数据类型, "non_null_count": 非空数,
                     "null_count": 空值数, "null_rate": 缺失率, "unique_count": 唯一值数}
                - "numeric_stats": 数值列统计信息
            - "file_info": 文件信息

    Example:
        >>> result = load_and_profile("data.csv")
        >>> print(result["profile"]["shape"])
        {'rows': 1000, 'columns': 10}
    """
    import pandas as pd
    import numpy as np

    if not os.path.isfile(csv_path):
        return {"error": f"文件不存在: {csv_path}"}

    file_size = os.path.getsize(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"读取CSV失败: {e}"}

    rows, cols = df.shape

    # 列信息
    columns_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        non_null_count = int(df[col].count())
        null_rate = round(null_count / rows * 100, 2) if rows > 0 else 0
        unique_count = int(df[col].nunique())

        columns_info.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_rate": null_rate,
            "unique_count": unique_count,
        })

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number])
    numeric_stats = {}
    for col in numeric_cols.columns:
        col_data = numeric_cols[col]
        numeric_stats[str(col)] = {
            "mean": round(float(col_data.mean()), 4) if not col_data.isnull().all() else None,
            "std": round(float(col_data.std()), 4) if not col_data.isnull().all() else None,
            "min": round(float(col_data.min()), 4) if not col_data.isnull().all() else None,
            "max": round(float(col_data.max()), 4) if not col_data.isnull().all() else None,
            "median": round(float(col_data.median()), 4) if not col_data.isnull().all() else None,
        }

    return {
        "data": df,
        "profile": {
            "shape": {"rows": int(rows), "columns": int(cols)},
            "columns": columns_info,
            "numeric_stats": numeric_stats,
        },
        "file_info": {
            "path": csv_path,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
        },
    }


# =============================================================================
# 2. 异常值检测
# =============================================================================
def detect_outliers(data, method="iqr", threshold=1.5):
    """
    检测数据中的异常值。

    Args:
        data: 可以是 pandas DataFrame 或字典 {列名: [数值列表]}。
        method (str): 检测方法，支持 "iqr"（四分位距）或 "zscore"（Z分数）。
        threshold (float): 阈值。IQR方法默认1.5（1.5倍IQR），Z-score方法默认3（3个标准差）。

    Returns:
        dict: 包含以下键的字典:
            - "method": 检测方法
            - "outliers": 异常值信息字典 {列名: {"indices": 索引列表, "values": 异常值列表, "count": 数量}}
            - "total_outliers": 总异常值数
            - "summary": 汇总信息

    Example:
        >>> result = detect_outliers(df, method="iqr", threshold=1.5)
    """
    import pandas as pd
    import numpy as np

    # 统一转换为DataFrame处理
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型，请传入DataFrame或字典"}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    total_outliers = 0

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            outlier_values = df.loc[outlier_mask, col].tolist()

        elif method == "zscore":
            mean = col_data.mean()
            std = col_data.std()

            if std == 0:
                continue

            z_scores = (df[col] - mean) / std
            outlier_mask = z_scores.abs() > threshold
            outlier_indices = df[outlier_mask].index.tolist()
            outlier_values = df.loc[outlier_mask, col].tolist()

        else:
            return {"error": f"不支持的检测方法: {method}"}

        count = len(outlier_indices)
        total_outliers += count

        if count > 0:
            outliers_info[str(col)] = {
                "indices": outlier_indices,
                "values": [round(float(v), 4) for v in outlier_values],
                "count": count,
            }

    return {
        "method": method,
        "outliers": outliers_info,
        "total_outliers": total_outliers,
        "summary": {
            "columns_checked": len(numeric_cols),
            "columns_with_outliers": len(outliers_info),
            "total_outliers": total_outliers,
        },
    }


# =============================================================================
# 3. 缺失值处理
# =============================================================================
def handle_missing(data, strategy="mean"):
    """
    处理数据中的缺失值。

    Args:
        data: pandas DataFrame 或字典 {列名: [值列表]}。
        strategy (str): 处理策略:
            - "drop": 删除含缺失值的行
            - "mean": 用均值填充（仅数值列）
            - "median": 用中位数填充（仅数值列）
            - "mode": 用众数填充
            - "interpolate": 线性插值（仅数值列）
            - "ffill": 前向填充
            - "bfill": 后向填充

    Returns:
        dict: 包含以下键的字典:
            - "data": 处理后的DataFrame
            - "strategy": 使用的策略
            - "before_missing": 处理前缺失值数
            - "after_missing": 处理后缺失值数
            - "filled_count": 填充的缺失值数

    Example:
        >>> result = handle_missing(df, strategy="mean")
    """
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    before_missing = int(df.isnull().sum().sum())

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "mode":
        for col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
    elif strategy == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        # 剩余的用前向/后向填充
        df = df.ffill().bfill()
    elif strategy == "ffill":
        df = df.ffill()
    elif strategy == "bfill":
        df = df.bfill()
    else:
        return {"error": f"不支持的策略: {strategy}"}

    after_missing = int(df.isnull().sum().sum())
    filled_count = before_missing - after_missing

    return {
        "data": df,
        "strategy": strategy,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "filled_count": filled_count,
    }


# =============================================================================
# 4. 去重
# =============================================================================
def remove_duplicates(data, subset=None):
    """
    移除数据中的重复行。

    Args:
        data: pandas DataFrame 或字典。
        subset (list, optional): 检查重复的列名列表。默认为None，检查所有列。

    Returns:
        dict: 包含以下键的字典:
            - "data": 去重后的DataFrame
            - "before_count": 去重前行数
            - "after_count": 去重后行数
            - "duplicates_removed": 移除的重复行数
            - "duplicate_rate": 重复率（百分比）

    Example:
        >>> result = remove_duplicates(df, subset=["id", "name"])
    """
    import pandas as pd

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    before_count = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    after_count = len(df)
    removed = before_count - after_count

    return {
        "data": df,
        "before_count": before_count,
        "after_count": after_count,
        "duplicates_removed": removed,
        "duplicate_rate": round(removed / before_count * 100, 2) if before_count > 0 else 0,
    }


# =============================================================================
# 5. 数据标准化
# =============================================================================
def normalize_data(data, method="minmax"):
    """
    对数值列进行标准化处理。

    Args:
        data: pandas DataFrame 或字典。
        method (str): 标准化方法:
            - "minmax": Min-Max标准化（缩放到0-1）
            - "zscore": Z-score标准化（均值0，标准差1）
            - "robust": Robust标准化（基于中位数和IQR）

    Returns:
        dict: 包含以下键的字典:
            - "data": 标准化后的DataFrame
            - "method": 使用的方法
            - "columns_normalized": 标准化的列列表
            - "params": 标准化参数（每列的min/max或mean/std）

    Example:
        >>> result = normalize_data(df, method="zscore")
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            "data": df,
            "method": method,
            "columns_normalized": [],
            "params": {},
        }

    params = {}

    if method == "minmax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for i, col in enumerate(numeric_cols):
            params[str(col)] = {
                "min": round(float(scaler.data_min_[i]), 4),
                "max": round(float(scaler.data_max_[i]), 4),
            }
    elif method == "zscore":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for i, col in enumerate(numeric_cols):
            params[str(col)] = {
                "mean": round(float(scaler.mean_[i]), 4),
                "std": round(float(scaler.scale_[i]), 4),
            }
    elif method == "robust":
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for i, col in enumerate(numeric_cols):
            params[str(col)] = {
                "median": round(float(scaler.center_[i]), 4),
                "iqr": round(float(scaler.scale_[i]), 4),
            }
    else:
        return {"error": f"不支持的方法: {method}"}

    return {
        "data": df,
        "method": method,
        "columns_normalized": [str(c) for c in numeric_cols],
        "params": params,
    }


# =============================================================================
# 6. 分类编码
# =============================================================================
def encode_categorical(data, method="label"):
    """
    对分类列进行编码。

    Args:
        data: pandas DataFrame 或字典。
        method (str): 编码方法:
            - "label": 标签编码（每类映射为整数）
            - "onehot": One-Hot编码（每类生成一个二进制列）
            - "target": 目标编码（需target_column参数，此处使用频率编码替代）

    Returns:
        dict: 包含以下键的字典:
            - "data": 编码后的DataFrame
            - "method": 使用的方法
            - "encoded_columns": 编码的列列表
            - "encoding_info": 编码信息（映射关系等）

    Example:
        >>> result = encode_categorical(df, method="onehot")
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    # 识别分类列
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_info = {}

    if method == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            # 处理缺失值
            non_null_mask = df[col].notnull()
            if non_null_mask.sum() > 0:
                df.loc[non_null_mask, col] = le.fit_transform(
                    df.loc[non_null_mask, col].astype(str)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
                encoding_info[str(col)] = {
                    "classes": list(le.classes_),
                    "mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
                }

    elif method == "onehot":
        for col in categorical_cols[:]:  # 复制列表因为在迭代中会修改
            dummies = pd.get_dummies(df[col], prefix=str(col), dummy_na=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            encoding_info[str(col)] = {
                "new_columns": list(dummies.columns),
            }

    elif method == "target":
        # 使用频率编码作为目标编码的简化版本
        for col in categorical_cols:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)
            encoding_info[str(col)] = {
                "frequency_map": {str(k): round(float(v), 4) for k, v in freq_map.items()},
            }
    else:
        return {"error": f"不支持的方法: {method}"}

    return {
        "data": df,
        "method": method,
        "encoded_columns": [str(c) for c in categorical_cols],
        "encoding_info": encoding_info,
    }


# =============================================================================
# 7. 数据质量报告
# =============================================================================
def generate_profile_report(data):
    """
    生成HTML格式的数据质量报告。

    Args:
        data: pandas DataFrame 或字典。

    Returns:
        dict: 包含以下键的字典:
            - "report": HTML格式报告文本
            - "summary": 报告摘要字典

    Example:
        >>> result = generate_profile_report(df)
        >>> with open("report.html", "w") as f:
        ...     f.write(result["report"])
    """
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = int(df.isnull().sum().sum())
    missing_rate = round(missing_cells / total_cells * 100, 2) if total_cells else 0
    duplicate_rows = int(df.duplicated().sum())

    # 生成HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html><head><meta charset='utf-8'>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
    html_parts.append(".header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }")
    html_parts.append(".section { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html_parts.append("table { border-collapse: collapse; width: 100%; }")
    html_parts.append("th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }")
    html_parts.append("th { background: #34495e; color: white; }")
    html_parts.append(".metric { display: inline-block; margin: 10px 20px; text-align: center; }")
    html_parts.append(".metric-value { font-size: 28px; font-weight: bold; color: #2c3e50; }")
    html_parts.append(".metric-label { font-size: 14px; color: #7f8c8d; }")
    html_parts.append(".bar { height: 20px; background: #e74c3c; border-radius: 4px; }")
    html_parts.append(".bar-container { background: #ecf0f1; border-radius: 4px; overflow: hidden; }")
    html_parts.append("</style></head><body>")

    # 标题
    html_parts.append(f"<div class='header'><h1>数据质量报告</h1><p>生成时间: {now}</p></div>")

    # 概要指标
    html_parts.append("<div class='section'><h2>概要</h2>")
    html_parts.append(f"<div class='metric'><div class='metric-value'>{rows}</div><div class='metric-label'>行数</div></div>")
    html_parts.append(f"<div class='metric'><div class='metric-value'>{cols}</div><div class='metric-label'>列数</div></div>")
    html_parts.append(f"<div class='metric'><div class='metric-value'>{missing_rate}%</div><div class='metric-label'>缺失率</div></div>")
    html_parts.append(f"<div class='metric'><div class='metric-value'>{duplicate_rows}</div><div class='metric-label'>重复行</div></div>")
    html_parts.append("</div>")

    # 列详情
    html_parts.append("<div class='section'><h2>列详情</h2>")
    html_parts.append("<table><tr><th>列名</th><th>类型</th><th>非空数</th><th>缺失数</th><th>缺失率</th><th>唯一值</th></tr>")

    column_stats = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        non_null = int(df[col].count())
        null_rate = round(null_count / rows * 100, 2) if rows > 0 else 0
        unique = int(df[col].nunique())

        html_parts.append(
            f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{non_null}</td>"
            f"<td>{null_count}</td><td>{null_rate}%</td><td>{unique}</td></tr>"
        )

        column_stats.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "non_null": non_null,
            "null_count": null_count,
            "null_rate": null_rate,
            "unique": unique,
        })

    html_parts.append("</table></div>")

    # 数值统计
    numeric_cols = df.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) > 0:
        html_parts.append("<div class='section'><h2>数值统计</h2>")
        html_parts.append("<table><tr><th>列名</th><th>均值</th><th>标准差</th><th>最小值</th><th>中位数</th><th>最大值</th></tr>")
        for col in numeric_cols.columns:
            col_data = numeric_cols[col]
            html_parts.append(
                f"<tr><td>{col}</td>"
                f"<td>{round(float(col_data.mean()), 4) if not col_data.isnull().all() else 'N/A'}</td>"
                f"<td>{round(float(col_data.std()), 4) if not col_data.isnull().all() else 'N/A'}</td>"
                f"<td>{round(float(col_data.min()), 4) if not col_data.isnull().all() else 'N/A'}</td>"
                f"<td>{round(float(col_data.median()), 4) if not col_data.isnull().all() else 'N/A'}</td>"
                f"<td>{round(float(col_data.max()), 4) if not col_data.isnull().all() else 'N/A'}</td></tr>"
            )
        html_parts.append("</table></div>")

    html_parts.append("</body></html>")

    return {
        "report": "\n".join(html_parts),
        "summary": {
            "rows": int(rows),
            "columns": int(cols),
            "missing_cells": missing_cells,
            "missing_rate": missing_rate,
            "duplicate_rows": duplicate_rows,
            "columns_info": column_stats,
        },
    }


# =============================================================================
# 8. 相关性分析
# =============================================================================
def correlation_analysis(data, method="pearson"):
    """
    分析数值列之间的相关性。

    Args:
        data: pandas DataFrame 或字典。
        method (str): 相关性计算方法:
            - "pearson": 皮尔逊相关系数（线性关系）
            - "spearman": 斯皮尔曼秩相关系数（单调关系）
            - "kendall": 肯德尔秩相关系数

    Returns:
        dict: 包含以下键的字典:
            - "correlation_matrix": 相关系数矩阵字典 {列名: {列名: 相关系数}}
            - "method": 使用的方法
            - "strong_correlations": 强相关对列表（|r| > 0.7）
            - "heatmap_data": 用于可视化的热力图数据

    Example:
        >>> result = correlation_analysis(df, method="pearson")
    """
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    numeric_cols = df.select_dtypes(include=[np.number])

    if len(numeric_cols.columns) < 2:
        return {
            "correlation_matrix": {},
            "method": method,
            "strong_correlations": [],
            "heatmap_data": [],
        }

    corr_matrix = numeric_cols.corr(method=method)

    # 转换为字典格式
    corr_dict = {}
    for col in corr_matrix.columns:
        corr_dict[str(col)] = {}
        for idx in corr_matrix.index:
            val = corr_matrix.loc[idx, col]
            corr_dict[str(col)][str(idx)] = round(float(val), 4) if not np.isnan(val) else None

    # 找出强相关对
    strong_corr = []
    cols_list = list(corr_matrix.columns)
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) > 0.7:
                strong_corr.append({
                    "col1": str(cols_list[i]),
                    "col2": str(cols_list[j]),
                    "correlation": round(float(val), 4),
                    "strength": "强正相关" if val > 0 else "强负相关",
                })

    # 热力图数据
    heatmap_data = []
    for i, row_name in enumerate(corr_matrix.index):
        for j, col_name in enumerate(corr_matrix.columns):
            val = corr_matrix.iloc[i, j]
            heatmap_data.append({
                "row": str(row_name),
                "col": str(col_name),
                "value": round(float(val), 4) if not np.isnan(val) else 0,
            })

    return {
        "correlation_matrix": corr_dict,
        "method": method,
        "strong_correlations": strong_corr,
        "heatmap_data": heatmap_data,
    }


# =============================================================================
# 9. 类型转换
# =============================================================================
def data_type_converter(data, conversions):
    """
    转换DataFrame列的数据类型。

    Args:
        data: pandas DataFrame 或字典。
        conversions (dict): 转换规则字典 {列名: 目标类型}。
            支持的类型: "int"/"float"/"str"/"bool"/"datetime"。
            datetime类型需要指定格式，如 "datetime:%Y-%m-%d"。

    Returns:
        dict: 包含以下键的字典:
            - "data": 转换后的DataFrame
            - "conversions": 转换结果列表，每项为 {"column": 列名, "from": 原类型, "to": 目标类型, "success": 是否成功}
            - "errors": 错误信息列表

    Example:
        >>> result = data_type_converter(df, {"price": "float", "date": "datetime:%Y-%m-%d"})
    """
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"error": "不支持的数据类型"}

    conversion_results = []
    errors = []

    for col, target_type in conversions.items():
        if col not in df.columns:
            errors.append(f"列 '{col}' 不存在")
            conversion_results.append({
                "column": col,
                "from": "N/A",
                "to": target_type,
                "success": False,
            })
            continue

        original_dtype = str(df[col].dtype)
        success = True

        try:
            if target_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target_type == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "str":
                df[col] = df[col].astype(str)
            elif target_type == "bool":
                df[col] = df[col].astype(bool)
            elif target_type.startswith("datetime:"):
                date_format = target_type.split(":", 1)[1]
                df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
            else:
                errors.append(f"列 '{col}': 不支持的类型 '{target_type}'")
                success = False
        except Exception as e:
            errors.append(f"列 '{col}' 转换失败: {e}")
            success = False

        conversion_results.append({
            "column": col,
            "from": original_dtype,
            "to": target_type,
            "success": success,
        })

    return {
        "data": df,
        "conversions": conversion_results,
        "errors": errors,
    }


# =============================================================================
# 10. 导出清洗后数据
# =============================================================================
def export_cleaned_data(data, format="csv", path="cleaned_data"):
    """
    将清洗后的数据导出为指定格式。

    Args:
        data: pandas DataFrame 或字典。
        format (str): 导出格式，支持 "csv"/"excel"/"json"/"parquet"。
        path (str): 输出文件路径（不含扩展名，会自动添加）。

    Returns:
        dict: 包含以下键的字典:
            - "success": 是否成功
            - "output_path": 输出文件路径
            - "format": 导出格式
            - "rows": 导出行数
            - "columns": 导出列数
            - "file_size": 文件大小（字节）
            - "errors": 错误信息列表

    Example:
        >>> result = export_cleaned_data(df, format="csv", path="output/clean")
    """
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {"success": False, "errors": ["不支持的数据类型"]}

    rows, cols = df.shape
    result = {
        "success": False,
        "output_path": "",
        "format": format,
        "rows": int(rows),
        "columns": int(cols),
        "file_size": 0,
        "errors": [],
    }

    # 确保目录存在
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        if format == "csv":
            output_path = f"{path}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif format == "excel":
            output_path = f"{path}.xlsx"
            df.to_excel(output_path, index=False, engine="openpyxl")
        elif format == "json":
            output_path = f"{path}.json"
            df.to_json(output_path, orient="records", force_ascii=False, indent=2)
        elif format == "parquet":
            output_path = f"{path}.parquet"
            df.to_parquet(output_path, index=False)
        else:
            result["errors"].append(f"不支持的格式: {format}")
            return result

        result["output_path"] = output_path
        result["file_size"] = os.path.getsize(output_path)
        result["success"] = True

    except Exception as e:
        result["errors"].append(f"导出失败: {e}")

    return result


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("数据清洗工具箱 (data-cleaner)")
    print("=" * 60)
    print("可用工具:")
    tools = [
        "1. load_and_profile          - 数据加载与画像",
        "2. detect_outliers           - 异常值检测",
        "3. handle_missing            - 缺失值处理",
        "4. remove_duplicates         - 去重",
        "5. normalize_data            - 标准化",
        "6. encode_categorical        - 分类编码",
        "7. generate_profile_report   - 数据质量报告",
        "8. correlation_analysis      - 相关性分析",
        "9. data_type_converter       - 类型转换",
        "10. export_cleaned_data      - 导出清洗后数据",
    ]
    for tool in tools:
        print(f"  {tool}")
    print("=" * 60)

    # 演示：缺失值处理
    print("\n演示 - 缺失值处理:")
    import pandas as pd
    import numpy as np
    demo_data = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": [10, np.nan, 30, 40, 50],
        "C": ["x", "y", "z", np.nan, "x"],
    })
    print("  原始数据:")
    print(demo_data.to_string())
    missing_result = handle_missing(demo_data, strategy="mean")
    print(f"\n  策略: {missing_result['strategy']}")
    print(f"  填充缺失值数: {missing_result['filled_count']}")
    print("  处理后数据:")
    print(missing_result["data"].to_string())
