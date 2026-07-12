"""
data-cleaner — 数据清洗工具箱

包含10个高级算法驱动的数据清洗与分析工具，全部使用Python标准库实现
（math, statistics, random, json, csv, re, collections, datetime等），
不依赖numpy/pandas/scikit-learn，所有算法从零实现。

功能列表:
  1. data_profiling                  — 数据画像（类型推断/统计/四分位/偏度峰度）
  2. outlier_detector_zscore          — 修改版Z-score异常检测（基于MAD）
  3. outlier_detector_isolation_forest — Isolation Forest完整实现
  4. missing_value_imputer            — 缺失值填充（KNN/均值/中位数/众数/回归）
  5. duplicate_detector              — 数据去重（精确+模糊匹配）
  6. data_normalizer                 — 数据标准化（5种方法）
  7. correlation_analyzer            — 相关性分析（Pearson/Spearman/Kendall+p值）
  8. data_type_converter             — 智能类型转换
  9. data_quality_scorer             — 数据质量评分模型（5维度）
 10. data_export_pipeline            — 数据导出管线（CSV/JSON/TSV/Excel XML+GZIP）

Author: github.com/wzx11223344
License: MIT
"""

import math
import csv
import json
import re
import gzip
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from io import StringIO, BytesIO


# ======================================================================
# 1. data_profiling — 数据画像分析
# ======================================================================

def data_profiling(dataset):
    """数据画像分析：对每列计算全面的统计指标。

    算法步骤:
      1. 推断每列的数据类型（int/float/bool/str/datetime）
      2. 计算基础统计：计数/唯一值数/缺失率/缺失数
      3. 对数值列：计算最小/最大/均值/标准差/方差/四分位数
      4. 计算偏度（Skewness）和峰度（Kurtosis）
      5. 对分类列：计算top-10高频值及其频率
      6. 汇总数据集级别的概况

    Args:
        dataset (list of dict): 数据集，每行是一个字典

    Returns:
        dict: {
            "row_count": int,
            "column_count": int,
            "columns": {
                col_name: {
                    "type": str,
                    "count": int,
                    "unique": int,
                    "missing": int,
                    "missing_rate": float,
                    "stats": {...},
                    "top_values": [(value, count), ...],
                }, ...
            },
            "summary": str,
        }
    """
    if not dataset:
        return {"row_count": 0, "column_count": 0, "columns": {}, "summary": "空数据集"}

    columns = {}
    all_keys = set()
    for row in dataset:
        all_keys.update(row.keys())

    for col in sorted(all_keys):
        values = [row.get(col) for row in dataset]
        col_info = _profile_column(col, values)
        columns[col] = col_info

    return {
        "row_count": len(dataset),
        "column_count": len(all_keys),
        "columns": columns,
        "summary": f"数据集包含 {len(dataset)} 行 x {len(all_keys)} 列",
    }


def _profile_column(col_name, values):
    """分析单列的统计信息。"""
    non_none = [v for v in values if v is not None and v != ""]
    missing_count = len(values) - len(non_none)
    unique_values = set(non_none)
    col_type = _infer_column_type(non_none)

    info = {
        "name": col_name,
        "type": col_type,
        "count": len(non_none),
        "unique": len(unique_values),
        "missing": missing_count,
        "missing_rate": round(missing_count / len(values) * 100, 2) if values else 0,
        "stats": {},
        "top_values": [],
    }

    # 数值型统计
    if col_type in ("int", "float"):
        numeric_vals = []
        for v in non_none:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_vals:
            n = len(numeric_vals)
            mean_val = sum(numeric_vals) / n
            min_val = min(numeric_vals)
            max_val = max(numeric_vals)

            # 方差和标准差
            variance = sum((x - mean_val) ** 2 for x in numeric_vals) / n if n > 0 else 0
            std_val = math.sqrt(variance)

            # 四分位数（使用线性插值法）
            sorted_vals = sorted(numeric_vals)
            q1 = _percentile(sorted_vals, 25)
            q2 = _percentile(sorted_vals, 50)
            q3 = _percentile(sorted_vals, 75)

            # 偏度（Skewness）— Pearson偏度系数
            if std_val > 0:
                skewness = (3 * (mean_val - q2)) / std_val
            else:
                skewness = 0

            # 峰度（Kurtosis）— 超额峰度
            if n > 2 and std_val > 0:
                kurt = (sum((x - mean_val) ** 4 for x in numeric_vals) / n) / (variance ** 2) - 3
            else:
                kurt = 0

            info["stats"] = {
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "variance": round(variance, 4),
                "q1": round(q1, 4),
                "median": round(q2, 4),
                "q3": round(q3, 4),
                "iqr": round(q3 - q1, 4),
                "range": round(max_val - min_val, 4),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurt, 4),
            }

    # Top-10 高频值
    counter = Counter(non_none)
    top = counter.most_common(10)
    info["top_values"] = [
        {"value": str(k), "count": c, "frequency": round(c / len(non_none) * 100, 2)}
        for k, c in top
    ]

    return info


def _infer_column_type(values):
    """推断列的数据类型。"""
    if not values:
        return "unknown"

    type_scores = {"int": 0, "float": 0, "bool": 0, "datetime": 0, "str": 0}
    for v in values:
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            type_scores["bool"] += 1
        elif isinstance(v, int):
            type_scores["int"] += 1
        elif isinstance(v, float):
            type_scores["float"] += 1
        elif isinstance(v, str):
            # 尝试解析为数值
            try:
                int(v)
                type_scores["int"] += 1
                continue
            except ValueError:
                pass
            try:
                float(v)
                type_scores["float"] += 1
                continue
            except ValueError:
                pass
            # 尝试解析为日期
            if _try_parse_datetime(v):
                type_scores["datetime"] += 1
            else:
                type_scores["str"] += 1

    return max(type_scores, key=type_scores.get)


def _percentile(sorted_data, p):
    """使用线性插值法计算百分位数。"""
    if not sorted_data:
        return 0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    rank = p / 100 * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _try_parse_datetime(s):
    """尝试解析字符串为日期时间。"""
    patterns = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{4}/\d{2}/\d{2}$",
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$",
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{4}\d{2}\d{2}$",
    ]
    for pat in patterns:
        if re.match(pat, str(s)):
            return True
    return False


# ======================================================================
# 2. outlier_detector_zscore — 修改版Z-score异常检测（基于MAD）
# ======================================================================

def outlier_detector_zscore(data, threshold=3):
    """基于MAD（中位数绝对偏差）的修改版Z-score异常检测。

    传统Z-score使用均值和标准差，但它们对异常值敏感。
    修改版Z-score使用中位数和MAD（Median Absolute Deviation），
    具有更好的鲁棒性。

    算法:
      1. 计算中位数 median
      2. 计算每个值与中位数的绝对偏差 |x_i - median|
      3. 计算MAD = median(|x_i - median|)
      4. 修改版Z-score: z_i = 0.6745 * (x_i - median) / MAD
      5. |z_i| > threshold 的点标记为异常值

    Args:
        data (list): 数值列表或字典列表
        threshold (float): 异常阈值（默认3，即3倍标准差）

    Returns:
        dict: {
            "total": int,
            "outliers": int,
            "outlier_rate": float,
            "median": float,
            "mad": float,
            "threshold": float,
            "details": [
                {"index": int, "value": float, "z_score": float, "is_outlier": bool}, ...
            ],
            "outlier_indices": [int, ...],
        }
    """
    # 提取数值
    if isinstance(data[0], dict):
        # 取第一个数值列
        values = []
        for item in data:
            for v in item.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
                    break
    else:
        values = [float(v) for v in data if v is not None and isinstance(v, (int, float))]

    if not values:
        return {"total": 0, "outliers": 0, "outlier_rate": 0, "details": []}

    n = len(values)
    sorted_vals = sorted(values)

    # 计算中位数
    median = _percentile(sorted_vals, 50)

    # 计算绝对偏差
    abs_devs = sorted(abs(v - median) for v in values)

    # 计算MAD
    mad = _percentile(abs_devs, 50)

    # 避免除零
    if mad == 0:
        # 使用标准差作为替代
        mean_val = sum(values) / n
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in values) / n) if n > 0 else 0
        if std_val == 0:
            mad = 1e-10
        else:
            # 使用标准差方式
            details = []
            outlier_indices = []
            for i, v in enumerate(values):
                z = (v - mean_val) / std_val
                is_outlier = abs(z) > threshold
                if is_outlier:
                    outlier_indices.append(i)
                details.append({
                    "index": i,
                    "value": round(v, 4),
                    "z_score": round(z, 4),
                    "is_outlier": is_outlier,
                })
            return {
                "total": n,
                "outliers": len(outlier_indices),
                "outlier_rate": round(len(outlier_indices) / n * 100, 2),
                "median": round(median, 4),
                "mad": 0,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "threshold": threshold,
                "method": "standard_zscore",
                "details": details,
                "outlier_indices": outlier_indices,
            }

    # 计算修改版Z-score
    details = []
    outlier_indices = []
    for i, v in enumerate(values):
        modified_z = 0.6745 * (v - median) / mad
        is_outlier = abs(modified_z) > threshold
        if is_outlier:
            outlier_indices.append(i)
        details.append({
            "index": i,
            "value": round(v, 4),
            "z_score": round(modified_z, 4),
            "is_outlier": is_outlier,
        })

    return {
        "total": n,
        "outliers": len(outlier_indices),
        "outlier_rate": round(len(outlier_indices) / n * 100, 2),
        "median": round(median, 4),
        "mad": round(mad, 4),
        "threshold": threshold,
        "method": "modified_zscore_mad",
        "details": details,
        "outlier_indices": outlier_indices,
    }


# ======================================================================
# 3. outlier_detector_isolation_forest — Isolation Forest完整实现
# ======================================================================

class _IsolationTree:
    """孤立树（Isolation Tree）节点。"""

    def __init__(self):
        self.size = 0
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.is_external = False
        self.external_path_length = 0


def _build_isolation_tree(data, features, depth, max_depth):
    """递归构建孤立树。

    算法: 随机选择一个特征和分割点，将数据分为左右两部分。
    """
    node = _IsolationTree()
    node.size = len(data)

    # 终止条件：达到最大深度或数据量<=1
    if depth >= max_depth or len(data) <= 1:
        node.is_external = True
        node.external_path_length = _expected_path_length(len(data))
        return node

    # 随机选择特征
    feature = random.choice(features)
    feature_values = [row[feature] for row in data if row.get(feature) is not None]

    if not feature_values:
        node.is_external = True
        node.external_path_length = _expected_path_length(len(data))
        return node

    min_val = min(feature_values)
    max_val = max(feature_values)

    if min_val == max_val:
        node.is_external = True
        node.external_path_length = _expected_path_length(len(data))
        return node

    # 随机选择分割点
    split_value = random.uniform(min_val, max_val)

    # 分割数据
    left_data = [row for row in data if row.get(feature) is not None and row[feature] < split_value]
    right_data = [row for row in data if row.get(feature) is not None and row[feature] >= split_value]

    if not left_data or not right_data:
        node.is_external = True
        node.external_path_length = _expected_path_length(len(data))
        return node

    node.split_feature = feature
    node.split_value = split_value
    node.left = _build_isolation_tree(left_data, features, depth + 1, max_depth)
    node.right = _build_isolation_tree(right_data, features, depth + 1, max_depth)
    return node


def _path_length(point, tree, depth=0):
    """计算数据点在孤立树中的路径长度。"""
    if tree.is_external:
        return depth + tree.external_path_length

    feature = tree.split_feature
    value = point.get(feature)

    if value is None:
        return depth + _expected_path_length(tree.size)

    if value < tree.split_value:
        return _path_length(point, tree.left, depth + 1)
    else:
        return _path_length(point, tree.right, depth + 1)


def _expected_path_length(n):
    """计算二叉搜索树的期望路径长度。

    c(n) = 2 * H(n-1) - 2*(n-1)/n
    其中 H(i) = ln(i) + 0.5772156649（欧拉常数）
    """
    if n <= 1:
        return 0
    if n == 2:
        return 1
    euler_constant = 0.5772156649015329
    h = math.log(n - 1) + euler_constant
    return 2 * h - 2 * (n - 1) / n


def outlier_detector_isolation_forest(data, n_trees=100, sample_size=256):
    """Isolation Forest异常检测算法完整实现。

    算法步骤:
      1. 从数据中随机采样构建n_trees棵孤立树
      2. 每棵树递归地随机选择特征和分割点进行分割
      3. 计算每个数据点在所有树中的平均路径长度 E(h)
      4. 异常得分: s = 2^(-E(h)/c(n))
         s接近1 → 异常值
         s接近0.5 → 正常值
      5. 得分>0.6的标记为异常

    Args:
        data (list of dict): 数据集
        n_trees (int): 树的数量
        sample_size (int): 每棵树的采样大小

    Returns:
        dict: {
            "total": int,
            "outliers": int,
            "outlier_rate": float,
            "scores": [{"index": int, "score": float, "is_outlier": bool}, ...],
            "n_trees": int,
            "sample_size": int,
        }
    """
    if not data:
        return {"total": 0, "outliers": 0, "outlier_rate": 0, "scores": []}

    # 提取数值特征
    features = set()
    for row in data:
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                features.add(k)

    features = list(features)
    if not features:
        return {"total": len(data), "outliers": 0, "outlier_rate": 0, "scores": [],
                "n_trees": n_trees, "sample_size": sample_size}

    # 限制采样大小
    actual_sample_size = min(sample_size, len(data))
    max_depth = math.ceil(math.log2(actual_sample_size)) if actual_sample_size > 1 else 1

    # 构建森林
    trees = []
    for _ in range(n_trees):
        # 随机采样（无放回）
        if len(data) > actual_sample_size:
            sample = random.sample(data, actual_sample_size)
        else:
            sample = data
        tree = _build_isolation_tree(sample, features, 0, max_depth)
        trees.append(tree)

    # 计算每个数据点的异常得分
    n = len(data)
    c_n = _expected_path_length(actual_sample_size)

    if c_n == 0:
        c_n = 1

    scores = []
    outlier_count = 0

    for i, point in enumerate(data):
        # 计算平均路径长度
        total_path = 0
        valid_trees = 0
        for tree in trees:
            path = _path_length(point, tree, 0)
            total_path += path
            valid_trees += 1

        avg_path = total_path / valid_trees if valid_trees > 0 else 0

        # 异常得分: s = 2^(-E(h)/c(n))
        anomaly_score = 2 ** (-avg_path / c_n)

        # 得分>0.6标记为异常
        is_outlier = anomaly_score > 0.6
        if is_outlier:
            outlier_count += 1

        scores.append({
            "index": i,
            "score": round(anomaly_score, 6),
            "avg_path_length": round(avg_path, 4),
            "is_outlier": is_outlier,
        })

    return {
        "total": n,
        "outliers": outlier_count,
        "outlier_rate": round(outlier_count / n * 100, 2) if n else 0,
        "scores": scores,
        "n_trees": n_trees,
        "sample_size": actual_sample_size,
    }


# ======================================================================
# 4. missing_value_imputer — 缺失值填充
# ======================================================================

def missing_value_imputer(data, strategy='knn', k=5):
    """缺失值填充器，支持多种填充策略。

    策略:
      - 'knn': K近邻填充，基于距离加权（适用于数值列）
      - 'mean': 均值填充
      - 'median': 中位数填充
      - 'mode': 众数填充
      - 'regression': 回归填充（简单线性回归）

    KNN算法:
      1. 对每个缺失值，找到k个最相似的完整记录
      2. 使用欧氏距离计算相似度
      3. 距离倒数加权平均填充

    回归填充算法:
      1. 以缺失列作为因变量，其他数值列作为自变量
      2. 在完整数据上训练简单线性回归
      3. 预测缺失值

    Args:
        data (list of dict): 含缺失值的数据集
        strategy (str): 填充策略
        k (int): KNN的k值

    Returns:
        dict: {
            "imputed_data": list of dict,
            "total_missing": int,
            "imputed": int,
            "strategy": str,
            "column_details": {col: {"missing_before": int, "imputed": int, "fill_value": ...}},
        }
    """
    if not data:
        return {"imputed_data": [], "total_missing": 0, "imputed": 0, "strategy": strategy}

    # 深拷贝数据
    imputed_data = [dict(row) for row in data]

    # 找出所有列
    all_columns = set()
    for row in imputed_data:
        all_columns.update(row.keys())

    # 识别数值列
    numeric_cols = []
    for col in all_columns:
        numeric_count = 0
        total_count = 0
        for row in imputed_data:
            val = row.get(col)
            if val is not None and val != "":
                total_count += 1
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numeric_count += 1
        if total_count > 0 and numeric_count / total_count > 0.7:
            numeric_cols.append(col)

    total_missing = 0
    total_imputed = 0
    column_details = {}

    for col in sorted(all_columns):
        missing_indices = []
        non_missing_values = []

        for i, row in enumerate(imputed_data):
            val = row.get(col)
            if val is None or val == "":
                missing_indices.append(i)
                total_missing += 1
            else:
                non_missing_values.append(val)

        if not missing_indices:
            column_details[col] = {"missing_before": 0, "imputed": 0}
            continue

        fill_value = None
        imputed_count = 0

        if strategy == 'mean' and col in numeric_cols:
            numeric_vals = [float(v) for v in non_missing_values if isinstance(v, (int, float))]
            if numeric_vals:
                fill_value = sum(numeric_vals) / len(numeric_vals)
            for idx in missing_indices:
                imputed_data[idx][col] = round(fill_value, 4) if fill_value else 0
                imputed_count += 1

        elif strategy == 'median' and col in numeric_cols:
            numeric_vals = sorted(float(v) for v in non_missing_values if isinstance(v, (int, float)))
            if numeric_vals:
                fill_value = _percentile(numeric_vals, 50)
            for idx in missing_indices:
                imputed_data[idx][col] = round(fill_value, 4) if fill_value else 0
                imputed_count += 1

        elif strategy == 'mode':
            counter = Counter(non_missing_values)
            if counter:
                fill_value = counter.most_common(1)[0][0]
            for idx in missing_indices:
                imputed_data[idx][col] = fill_value
                imputed_count += 1

        elif strategy == 'knn' and col in numeric_cols:
            # KNN填充
            fill_values = _knn_impute(imputed_data, col, missing_indices, numeric_cols, k)
            for idx, val in zip(missing_indices, fill_values):
                imputed_data[idx][col] = round(val, 4) if val is not None else 0
                imputed_count += 1
            fill_value = "(multiple KNN values)"

        elif strategy == 'regression' and col in numeric_cols:
            # 回归填充
            fill_values = _regression_impute(imputed_data, col, missing_indices, numeric_cols)
            for idx, val in zip(missing_indices, fill_values):
                imputed_data[idx][col] = round(val, 4) if val is not None else 0
                imputed_count += 1
            fill_value = "(multiple regression values)"

        else:
            # 默认使用众数
            counter = Counter(non_missing_values)
            if counter:
                fill_value = counter.most_common(1)[0][0]
            for idx in missing_indices:
                imputed_data[idx][col] = fill_value
                imputed_count += 1

        total_imputed += imputed_count
        column_details[col] = {
            "missing_before": len(missing_indices),
            "imputed": imputed_count,
            "fill_value": fill_value,
        }

    return {
        "imputed_data": imputed_data,
        "total_missing": total_missing,
        "imputed": total_imputed,
        "strategy": strategy,
        "column_details": column_details,
    }


def _knn_impute(data, target_col, missing_indices, numeric_cols, k):
    """KNN缺失值填充，使用距离倒数加权。"""
    fill_values = []

    # 准备完整记录作为参考集
    reference_set = []
    for i, row in enumerate(data):
        if i not in missing_indices and row.get(target_col) is not None:
            features = []
            for col in numeric_cols:
                if col != target_col:
                    val = row.get(col)
                    if isinstance(val, (int, float)):
                        features.append(float(val))
                    else:
                        features.append(0.0)
            reference_set.append({
                "index": i,
                "features": features,
                "target": float(row[target_col]),
            })

    if not reference_set:
        return [0.0] * len(missing_indices)

    # 计算参考集特征的标准化范围
    n_features = len(reference_set[0]["features"]) if reference_set else 0
    if n_features == 0:
        # 没有其他数值列，用均值
        mean_val = sum(r["target"] for r in reference_set) / len(reference_set)
        return [mean_val] * len(missing_indices)

    feature_ranges = []
    for f in range(n_features):
        vals = [r["features"][f] for r in reference_set]
        min_v, max_v = min(vals), max(vals)
        feature_ranges.append(max_v - min_v if max_v != min_v else 1)

    for idx in missing_indices:
        row = data[idx]
        # 计算缺失行特征
        query_features = []
        for col in numeric_cols:
            if col != target_col:
                val = row.get(col)
                if isinstance(val, (int, float)):
                    query_features.append(float(val))
                else:
                    query_features.append(0.0)

        # 计算到所有参考点的距离
        distances = []
        for ref in reference_set:
            dist = 0
            for f in range(n_features):
                diff = (query_features[f] - ref["features"][f]) / feature_ranges[f]
                dist += diff ** 2
            dist = math.sqrt(dist) if dist > 0 else 1e-10
            distances.append((dist, ref["target"]))

        # 取k个最近邻
        distances.sort(key=lambda x: x[0])
        k_actual = min(k, len(distances))

        # 距离倒数加权平均
        total_weight = 0
        weighted_sum = 0
        for dist, target in distances[:k_actual]:
            weight = 1.0 / (dist + 1e-10)
            total_weight += weight
            weighted_sum += target * weight

        fill_value = weighted_sum / total_weight if total_weight > 0 else 0
        fill_values.append(fill_value)

    return fill_values


def _regression_impute(data, target_col, missing_indices, numeric_cols):
    """简单线性回归填充（使用最小二乘法）。"""
    # 准备训练数据
    X = []
    y = []
    predictor_cols = [c for c in numeric_cols if c != target_col]

    if not predictor_cols:
        # 没有预测变量，用均值
        all_vals = []
        for row in data:
            val = row.get(target_col)
            if isinstance(val, (int, float)) and val is not None:
                all_vals.append(float(val))
        mean_val = sum(all_vals) / len(all_vals) if all_vals else 0
        return [mean_val] * len(missing_indices)

    for row in data:
        val = row.get(target_col)
        if isinstance(val, (int, float)) and val is not None:
            features = []
            for col in predictor_cols:
                v = row.get(col)
                if isinstance(v, (int, float)):
                    features.append(float(v))
                else:
                    features.append(0.0)
            X.append(features)
            y.append(float(val))

    if len(X) < 2:
        mean_val = sum(y) / len(y) if y else 0
        return [mean_val] * len(missing_indices)

    # 最小二乘法: w = (X^T X)^{-1} X^T y
    # 使用简单的正规方程
    n_features = len(predictor_cols)

    # 构建矩阵 X^T X 和 X^T y
    XtX = [[0.0] * n_features for _ in range(n_features)]
    Xty = [0.0] * n_features

    for i in range(len(X)):
        for j in range(n_features):
            for l in range(n_features):
                XtX[j][l] += X[i][j] * X[i][l]
            Xty[j] += X[i][j] * y[i]

    # 加正则化防止奇异
    for i in range(n_features):
        XtX[i][i] += 1e-8

    # 解线性方程组（使用高斯消元法）
    weights = _gaussian_elimination(XtX, Xty)
    bias = sum(y) / len(y) - sum(weights[j] * (sum(X[i][j] for i in range(len(X))) / len(X))
                for j in range(n_features))

    # 预测缺失值
    fill_values = []
    for idx in missing_indices:
        row = data[idx]
        prediction = bias
        for j, col in enumerate(predictor_cols):
            v = row.get(col)
            if isinstance(v, (int, float)):
                prediction += weights[j] * float(v)
        fill_values.append(prediction)

    return fill_values


def _gaussian_elimination(A, b):
    """高斯消元法解线性方程组 Ax = b。"""
    n = len(A)
    # 构建增广矩阵
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    # 前向消元
    for col in range(n):
        # 找主元
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-15:
            continue

        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-15:
            x[i] = 0
            continue
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


# ======================================================================
# 5. duplicate_detector — 数据去重
# ======================================================================

def duplicate_detector(data, method='fuzzy'):
    """数据去重检测器，支持精确匹配和模糊匹配。

    精确匹配:
      - 完全相同的记录（所有字段值相同）

    模糊匹配:
      - 字符串字段：使用编辑距离（Levenshtein）计算相似度
      - 数值字段：允许指定容差范围内视为相同
      - 综合相似度 = 各字段相似度的加权平均
      - 相似度>0.85的记录对标记为重复

    Args:
        data (list of dict): 数据集
        method (str): 'exact' 或 'fuzzy'

    Returns:
        dict: {
            "total_records": int,
            "duplicate_pairs": int,
            "unique_records": int,
            "duplicates": [
                {"record1_index": int, "record2_index": int,
                 "similarity": float, "matching_fields": [str, ...]}, ...
            ],
            "duplicate_groups": [[int, ...], ...],
        }
    """
    n = len(data)
    if n < 2:
        return {"total_records": n, "duplicate_pairs": 0, "unique_records": n,
                "duplicates": [], "duplicate_groups": []}

    duplicate_pairs = []
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    all_columns = sorted(all_columns)

    # 构建记录的规范化表示
    normalized = []
    for row in data:
        norm_row = {}
        for col in all_columns:
            val = row.get(col)
            if val is None:
                norm_row[col] = ""
            elif isinstance(val, str):
                norm_row[col] = val.strip().lower()
            else:
                norm_row[col] = val
        normalized.append(norm_row)

    if method == 'exact':
        # 精确匹配：使用哈希
        seen = {}
        for i in range(n):
            # 创建记录的哈希键
            key_parts = []
            for col in all_columns:
                val = normalized[i].get(col)
                key_parts.append(f"{col}={val}")
            key = "|".join(key_parts)

            if key in seen:
                j = seen[key]
                duplicate_pairs.append({
                    "record1_index": j,
                    "record2_index": i,
                    "similarity": 1.0,
                    "matching_fields": all_columns[:],
                })
            else:
                seen[key] = i
    else:
        # 模糊匹配：两两比较
        for i in range(n):
            for j in range(i + 1, n):
                sim, matching = _record_similarity(normalized[i], normalized[j], all_columns)
                if sim >= 0.85:
                    duplicate_pairs.append({
                        "record1_index": i,
                        "record2_index": j,
                        "similarity": round(sim, 4),
                        "matching_fields": matching,
                    })

    # 使用并查集构建重复组
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for pair in duplicate_pairs:
        union(pair["record1_index"], pair["record2_index"])

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    duplicate_groups = [sorted(g) for g in groups.values() if len(g) > 1]

    unique_count = n - sum(len(g) - 1 for g in duplicate_groups)

    return {
        "total_records": n,
        "duplicate_pairs": len(duplicate_pairs),
        "unique_records": unique_count,
        "duplicates": duplicate_pairs,
        "duplicate_groups": duplicate_groups,
    }


def _record_similarity(r1, r2, columns):
    """计算两条记录的相似度。"""
    total_sim = 0
    matching = []
    for col in columns:
        v1 = r1.get(col)
        v2 = r2.get(col)

        if v1 == v2:
            total_sim += 1.0
            matching.append(col)
        elif isinstance(v1, str) and isinstance(v2, str) and v1 and v2:
            # 字符串编辑距离
            dist = _levenshtein(v1, v2)
            max_len = max(len(v1), len(v2))
            sim = 1 - dist / max_len if max_len > 0 else 0
            if sim >= 0.85:
                matching.append(col)
            total_sim += sim
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # 数值容差
            if v1 == 0 and v2 == 0:
                sim = 1.0
            else:
                diff = abs(v1 - v2)
                max_val = max(abs(v1), abs(v2))
                sim = 1 - diff / max_val if max_val > 0 else 1
            if sim >= 0.85:
                matching.append(col)
            total_sim += sim
        else:
            total_sim += 0

    return total_sim / len(columns) if columns else 0, matching


def _levenshtein(s1, s2):
    """编辑距离（Levenshtein距离）— 使用滚动数组优化的动态规划。"""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ======================================================================
# 6. data_normalizer — 数据标准化
# ======================================================================

def data_normalizer(data, method='minmax'):
    """数据标准化，支持5种方法。

    方法:
      - 'minmax': Min-Max归一化 → [0, 1]
      - 'zscore': Z-Score标准化 → 均值0, 标准差1
      - 'robust': Robust Scaling → 基于IQR，中位数0, IQR=1
      - 'decimal': Decimal Scaling → 除以10的幂次使|v|<1
      - 'log': Log Scaling → log(1+v)

    Args:
        data (list): 数值列表 或 list of dict
        method (str): 标准化方法

    Returns:
        dict: {
            "normalized_data": list,
            "method": str,
            "params": {...},
        }
    """
    # 提取数值
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # 字典列表：对每个数值列分别标准化
        return _normalize_dict_list(data, method)
    else:
        values = [float(v) for v in data if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return _normalize_values(values, method)


def _normalize_values(values, method):
    """对数值列表进行标准化。"""
    if not values:
        return {"normalized_data": [], "method": method, "params": {}}

    n = len(values)
    normalized = []

    if method == 'minmax':
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        normalized = [(v - min_val) / range_val for v in values]
        params = {"min": min_val, "max": max_val, "range": range_val}

    elif method == 'zscore':
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_val = math.sqrt(variance) if variance > 0 else 1
        normalized = [(v - mean_val) / std_val for v in values]
        params = {"mean": mean_val, "std": std_val}

    elif method == 'robust':
        sorted_vals = sorted(values)
        median = _percentile(sorted_vals, 50)
        q1 = _percentile(sorted_vals, 25)
        q3 = _percentile(sorted_vals, 75)
        iqr = q3 - q1 if q3 != q1 else 1
        normalized = [(v - median) / iqr for v in values]
        params = {"median": median, "q1": q1, "q3": q3, "iqr": iqr}

    elif method == 'decimal':
        max_abs = max(abs(v) for v in values) if values else 1
        if max_abs == 0:
            j = 0
        else:
            j = math.ceil(math.log10(max_abs))
        divisor = 10 ** j if j > 0 else 1
        normalized = [v / divisor for v in values]
        params = {"max_abs": max_abs, "j": j, "divisor": divisor}

    elif method == 'log':
        # log(1+v) 变换，处理负值
        min_val = min(values)
        offset = abs(min_val) + 1 if min_val < 0 else 0
        normalized = [math.log(1 + v + offset) for v in values]
        params = {"offset": offset, "log_base": "e"}

    else:
        normalized = values[:]
        params = {}

    return {
        "normalized_data": [round(v, 6) for v in normalized],
        "method": method,
        "params": params,
    }


def _normalize_dict_list(data, method):
    """对字典列表的每个数值列进行标准化。"""
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())

    numeric_cols = []
    for col in all_columns:
        count = sum(1 for row in data if isinstance(row.get(col), (int, float)) and not isinstance(row.get(col), bool))
        if count > len(data) * 0.5:
            numeric_cols.append(col)

    result_data = [dict(row) for row in data]
    params = {}

    for col in numeric_cols:
        values = [float(row[col]) for row in data if isinstance(row.get(col), (int, float)) and not isinstance(row.get(col), bool)]
        result = _normalize_values(values, method)

        # 填回数据
        val_idx = 0
        for i in range(len(data)):
            if isinstance(result_data[i].get(col), (int, float)) and not isinstance(result_data[i].get(col), bool):
                result_data[i][col] = result["normalized_data"][val_idx]
                val_idx += 1

        params[col] = result["params"]

    return {
        "normalized_data": result_data,
        "method": method,
        "params": params,
        "normalized_columns": numeric_cols,
    }


# ======================================================================
# 7. correlation_analyzer — 相关性分析
# ======================================================================

def correlation_analyzer(data, method='pearson'):
    """相关性分析：实现Pearson/Spearman/Kendall三种相关系数。

    Pearson相关系数:
      衡量线性相关程度
      r = Σ(xi-x̄)(yi-ȳ) / √(Σ(xi-x̄)² · Σ(yi-ȳ)²)

    Spearman秩相关系数:
      非参数检验，衡量单调关系
      将值转为秩次后计算Pearson相关

    Kendall Tau相关系数:
      基于一致对和非一致对的数量
      tau = (一致对数 - 非一致对数) / (n*(n-1)/2)

    p值检验:
      使用t分布近似计算双尾p值
      t = r * √(n-2) / √(1-r²)
      p = 2 * (1 - CDF(|t|, df=n-2))

    Args:
        data (list of dict): 数据集
        method (str): 'pearson', 'spearman', 或 'kendall'

    Returns:
        dict: {
            "method": str,
            "columns": [str, ...],
            "matrix": [[float, ...], ...],
            "p_values": [[float, ...], ...],
            "significant_pairs": [{col1, col2, r, p_value}, ...],
        }
    """
    # 提取数值列
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())

    numeric_cols = []
    col_data = {}
    for col in sorted(all_columns):
        values = []
        for row in data:
            v = row.get(col)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(float(v))
        if len(values) > len(data) * 0.5 and len(values) >= 3:
            numeric_cols.append(col)
            col_data[col] = values

    n_cols = len(numeric_cols)
    if n_cols < 2:
        return {"method": method, "columns": numeric_cols, "matrix": [],
                "p_values": [], "significant_pairs": []}

    # 确保所有列长度一致（取最小长度）
    min_len = min(len(col_data[c]) for c in numeric_cols)
    for col in numeric_cols:
        col_data[col] = col_data[col][:min_len]

    n = min_len
    matrix = [[0.0] * n_cols for _ in range(n_cols)]
    p_values = [[0.0] * n_cols for _ in range(n_cols)]

    significant_pairs = []

    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                matrix[i][j] = 1.0
                p_values[i][j] = 0.0
                continue

            x = col_data[numeric_cols[i]]
            y = col_data[numeric_cols[j]]

            if method == 'pearson':
                r = _pearson_correlation(x, y)
            elif method == 'spearman':
                r = _spearman_correlation(x, y)
            elif method == 'kendall':
                r = _kendall_tau(x, y)
            else:
                r = _pearson_correlation(x, y)

            matrix[i][j] = round(r, 4)
            p_val = _correlation_p_value(r, n)
            p_values[i][j] = round(p_val, 4)

            if i < j and abs(r) > 0.3 and p_val < 0.05:
                significant_pairs.append({
                    "col1": numeric_cols[i],
                    "col2": numeric_cols[j],
                    "r": round(r, 4),
                    "p_value": round(p_val, 4),
                    "strength": _correlation_strength(r),
                })

    return {
        "method": method,
        "columns": numeric_cols,
        "matrix": matrix,
        "p_values": p_values,
        "significant_pairs": significant_pairs,
    }


def _pearson_correlation(x, y):
    """计算Pearson相关系数。"""
    n = len(x)
    if n != len(y) or n == 0:
        return 0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return 0

    return numerator / denominator


def _spearman_correlation(x, y):
    """计算Spearman秩相关系数。"""
    rank_x = _compute_ranks(x)
    rank_y = _compute_ranks(y)
    return _pearson_correlation(rank_x, rank_y)


def _compute_ranks(values):
    """计算值的秩次（处理并列值用平均秩）。"""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # 平均秩
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def _kendall_tau(x, y):
    """计算Kendall Tau相关系数。

    比较所有数对，计算一致对（同序）和非一致对（反序）。
    tau = (一致对 - 非一致对) / 总对数
    """
    n = len(x)
    if n != len(y) or n < 2:
        return 0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            # 比较两个变量在i和j的顺序
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1

    total = n * (n - 1) / 2
    if total == 0:
        return 0

    return (concordant - discordant) / total


def _correlation_p_value(r, n):
    """计算相关系数的p值（t分布近似）。

    t = r * √(n-2) / √(1-r²)
    """
    if n <= 2:
        return 1.0

    df = n - 2
    if abs(r) >= 1.0:
        return 0.0

    t_stat = r * math.sqrt(df) / math.sqrt(1 - r * r)

    # 使用正态分布近似t分布（大样本）
    # p = 2 * (1 - Φ(|t|))
    p_value = 2 * (1 - _normal_cdf(abs(t_stat), df))

    return min(max(p_value, 0.0), 1.0)


def _normal_cdf(x, df=30):
    """标准正态分布的累积分布函数近似值。

    使用Laplace近似公式：
    Φ(x) ≈ 1 - 0.5 * exp(-(x²) / 2)  (当x较大时)
    """
    # 使用有理函数近似
    # Abramowitz & Stegun 公式 7.1.26
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    if x < 0:
        return 1 - _normal_cdf(-x, df)

    t = 1.0 / (1.0 + b0 * x)
    phi = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * (
        b1 * t + b2 * t ** 2 + b3 * t ** 3 + b4 * t ** 4 + b5 * t ** 5
    )
    return phi


def _correlation_strength(r):
    """描述相关强度。"""
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "强相关"
    elif abs_r >= 0.5:
        return "中等相关"
    elif abs_r >= 0.3:
        return "弱相关"
    else:
        return "极弱相关或无关"


# ======================================================================
# 8. data_type_converter — 智能类型转换
# ======================================================================

def data_type_converter(data, target_types=None):
    """智能类型转换器。

    算法:
      1. 如果提供了target_types，按照指定类型转换
      2. 如果未提供，自动推断最佳类型：
         - 检查是否为布尔值 (true/false/yes/no/1/0)
         - 检查是否为整数
         - 检查是否为浮点数
         - 检查是否为日期时间
         - 检查是否为分类（唯一值数<20）
      3. 处理格式不一致（如"1,234.56"→1234.56）
      4. 保留无法转换的原始值

    Args:
        data (list of dict): 数据集
        target_types (dict): {column: 'int'/'float'/'bool'/'datetime'/'str'}

    Returns:
        dict: {
            "converted_data": list of dict,
            "inferred_types": {col: type_str},
            "conversion_stats": {col: {"converted": int, "failed": int, "rate": float}},
        }
    """
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    all_columns = sorted(all_columns)

    # 推断类型或使用指定类型
    if target_types:
        types = target_types
    else:
        types = {}
        for col in all_columns:
            types[col] = _infer_best_type(data, col)

    converted_data = []
    conversion_stats = {col: {"converted": 0, "failed": 0, "rate": 0} for col in all_columns}

    for row in data:
        new_row = {}
        for col in all_columns:
            val = row.get(col)
            target = types.get(col, "str")
            converted, success = _convert_value(val, target)
            new_row[col] = converted
            if success:
                conversion_stats[col]["converted"] += 1
            elif val is not None and val != "":
                conversion_stats[col]["failed"] += 1
        converted_data.append(new_row)

    # 计算转换率
    for col in all_columns:
        total = conversion_stats[col]["converted"] + conversion_stats[col]["failed"]
        conversion_stats[col]["rate"] = (
            round(conversion_stats[col]["converted"] / total * 100, 2) if total else 100
        )

    return {
        "converted_data": converted_data,
        "inferred_types": types,
        "conversion_stats": conversion_stats,
    }


def _infer_best_type(data, col):
    """推断列的最佳数据类型。"""
    values = [row.get(col) for row in data if row.get(col) is not None and row.get(col) != ""]

    if not values:
        return "str"

    # 检查布尔
    bool_count = 0
    for v in values:
        if isinstance(v, bool):
            bool_count += 1
        elif isinstance(v, str) and v.lower() in ("true", "false", "yes", "no", "1", "0"):
            bool_count += 1
    if bool_count == len(values):
        return "bool"

    # 检查整数
    int_count = 0
    for v in values:
        if isinstance(v, int) and not isinstance(v, bool):
            int_count += 1
        elif isinstance(v, str) and _try_parse_int(v) is not None:
            int_count += 1
    if int_count == len(values):
        return "int"

    # 检查浮点
    float_count = 0
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            float_count += 1
        elif isinstance(v, str) and _try_parse_float(v) is not None:
            float_count += 1
    if float_count == len(values):
        return "float"

    # 检查日期
    date_count = 0
    for v in values:
        if isinstance(v, str) and _try_parse_date(v):
            date_count += 1
    if date_count / len(values) > 0.8:
        return "datetime"

    # 检查分类（唯一值少）
    unique = set(values)
    if len(unique) < 20:
        return "category"

    return "str"


def _try_parse_int(s):
    """尝试解析整数，处理千分位分隔符。"""
    s = str(s).strip().replace(",", "").replace(" ", "")
    try:
        return int(s)
    except ValueError:
        return None


def _try_parse_float(s):
    """尝试解析浮点数，处理千分位和百分号。"""
    s = str(s).strip().replace(",", "").replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
        try:
            return float(s) / 100
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _try_parse_date(s):
    """尝试解析日期字符串。"""
    s = str(s).strip()
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%Y%m%d", "%d-%m-%Y", "%d.%m.%Y",
    ]
    for fmt in formats:
        try:
            datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False


def _convert_value(val, target_type):
    """将值转换为目标类型。"""
    if val is None or val == "":
        return val, False

    if target_type == "int":
        parsed = _try_parse_int(val)
        if parsed is not None:
            return parsed, True
        return val, False

    elif target_type == "float":
        parsed = _try_parse_float(val)
        if parsed is not None:
            return parsed, True
        return val, False

    elif target_type == "bool":
        if isinstance(val, bool):
            return val, True
        s = str(val).lower().strip()
        if s in ("true", "yes", "1", "t", "y"):
            return True, True
        elif s in ("false", "no", "0", "f", "n"):
            return False, True
        return val, False

    elif target_type == "datetime":
        if isinstance(val, datetime):
            return val, True
        s = str(val).strip()
        formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%Y%m%d", "%d-%m-%Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt), True
            except ValueError:
                continue
        return val, False

    elif target_type == "category":
        return str(val), True

    else:
        return str(val) if not isinstance(val, str) else val, False


# ======================================================================
# 9. data_quality_scorer — 数据质量评分模型
# ======================================================================

def data_quality_scorer(dataset):
    """数据质量评分模型：评估5个维度。

    5个维度（每个0-100分，综合得分=加权平均）:
      1. 完整性(Completeness) 30%: 非空值比例
      2. 唯一性(Uniqueness) 20%: 主键去重率
      3. 一致性(Consistency) 20%: 数据类型一致性
      4. 准确性(Accuracy) 20%: 异常值比例
      5. 时效性(Timeliness) 10%: 数据新鲜度（如有日期列）

    Args:
        dataset (list of dict): 数据集

    Returns:
        dict: {
            "overall_score": int,
            "grade": str,
            "dimensions": {
                "completeness": {"score": int, "details": ...},
                "uniqueness": {"score": int, "details": ...},
                "consistency": {"score": int, "details": ...},
                "accuracy": {"score": int, "details": ...},
                "timeliness": {"score": int, "details": ...},
            },
            "recommendations": [str, ...],
        }
    """
    if not dataset:
        return {"overall_score": 0, "grade": "F", "dimensions": {}, "recommendations": ["数据集为空"]}

    n_rows = len(dataset)
    all_columns = set()
    for row in dataset:
        all_columns.update(row.keys())
    all_columns = sorted(all_columns)
    n_cols = len(all_columns)

    # 1. 完整性
    total_cells = n_rows * n_cols
    empty_cells = 0
    for row in dataset:
        for col in all_columns:
            val = row.get(col)
            if val is None or val == "":
                empty_cells += 1
    completeness_score = round((total_cells - empty_cells) / total_cells * 100, 2)

    # 2. 唯一性
    row_hashes = set()
    for row in dataset:
        row_hash = tuple(sorted((k, str(v)) for k, v in row.items()))
        row_hashes.add(row_hash)
    uniqueness_score = round(len(row_hashes) / n_rows * 100, 2)

    # 3. 一致性
    consistency_scores = {}
    for col in all_columns:
        types = set()
        for row in dataset:
            val = row.get(col)
            if val is not None and val != "":
                if isinstance(val, bool):
                    types.add("bool")
                elif isinstance(val, int):
                    types.add("int")
                elif isinstance(val, float):
                    types.add("float")
                elif isinstance(val, str):
                    types.add("str")
                else:
                    types.add("other")
        consistency_scores[col] = 1 / len(types) if types else 1
    avg_consistency = sum(consistency_scores.values()) / n_cols if n_cols else 1
    consistency_score = round(avg_consistency * 100, 2)

    # 4. 准确性（异常值检测）
    total_outliers = 0
    total_numeric = 0
    for col in all_columns:
        values = [row.get(col) for row in dataset
                  if isinstance(row.get(col), (int, float)) and not isinstance(row.get(col), bool)]
        if len(values) > 3:
            total_numeric += len(values)
            result = outlier_detector_zscore(values, threshold=3)
            total_outliers += result["outliers"]
    accuracy_score = round((1 - total_outliers / total_numeric) * 100, 2) if total_numeric > 0 else 100

    # 5. 时效性
    timeliness_score = 100
    date_cols = []
    for col in all_columns:
        for row in dataset:
            val = row.get(col)
            if isinstance(val, str) and _try_parse_date(val):
                date_cols.append(col)
                break
            elif isinstance(val, datetime):
                date_cols.append(col)
                break

    if date_cols:
        # 找最近的日期
        max_date = None
        for col in date_cols:
            for row in dataset:
                val = row.get(col)
                if isinstance(val, datetime):
                    if max_date is None or val > max_date:
                        max_date = val
                elif isinstance(val, str):
                    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
                        try:
                            dt = datetime.strptime(val, fmt)
                            if max_date is None or dt > max_date:
                                max_date = dt
                            break
                        except ValueError:
                            continue
        if max_date:
            days_old = (datetime.now() - max_date).days
            if days_old <= 30:
                timeliness_score = 100
            elif days_old <= 90:
                timeliness_score = 80
            elif days_old <= 180:
                timeliness_score = 60
            elif days_old <= 365:
                timeliness_score = 40
            else:
                timeliness_score = 20
    else:
        timeliness_score = 50  # 没有日期列，给中等分

    # 综合得分（加权平均）
    weights = {
        "completeness": 0.30,
        "uniqueness": 0.20,
        "consistency": 0.20,
        "accuracy": 0.20,
        "timeliness": 0.10,
    }

    scores = {
        "completeness": completeness_score,
        "uniqueness": uniqueness_score,
        "consistency": consistency_score,
        "accuracy": accuracy_score,
        "timeliness": timeliness_score,
    }

    overall = sum(scores[k] * weights[k] for k in weights)

    # 评级
    if overall >= 90:
        grade = "A"
    elif overall >= 80:
        grade = "B"
    elif overall >= 70:
        grade = "C"
    elif overall >= 60:
        grade = "D"
    else:
        grade = "F"

    # 生成建议
    recommendations = []
    if completeness_score < 80:
        recommendations.append(f"数据完整性不足（{completeness_score}%），建议检查缺失值填充策略")
    if uniqueness_score < 95:
        recommendations.append(f"数据唯一性偏低（{uniqueness_score}%），建议检查重复记录")
    if consistency_score < 80:
        recommendations.append(f"数据类型一致性不足（{consistency_score}%），建议统一数据格式")
    if accuracy_score < 90:
        recommendations.append(f"数据准确性需关注（{accuracy_score}%），建议处理异常值")
    if timeliness_score < 60:
        recommendations.append(f"数据时效性不足（{timeliness_score}%），建议更新数据")
    if not recommendations:
        recommendations.append("数据质量良好，无需特殊处理")

    return {
        "overall_score": round(overall, 2),
        "grade": grade,
        "dimensions": {
            "completeness": {
                "score": completeness_score,
                "weight": "30%",
                "total_cells": total_cells,
                "empty_cells": empty_cells,
                "details": "非空值比例",
            },
            "uniqueness": {
                "score": uniqueness_score,
                "weight": "20%",
                "total_records": n_rows,
                "unique_records": len(row_hashes),
                "details": "记录去重率",
            },
            "consistency": {
                "score": consistency_score,
                "weight": "20%",
                "column_consistency": {col: round(v, 4) for col, v in consistency_scores.items()},
                "details": "数据类型一致性",
            },
            "accuracy": {
                "score": accuracy_score,
                "weight": "20%",
                "total_numeric": total_numeric,
                "outliers": total_outliers,
                "details": "非异常值比例",
            },
            "timeliness": {
                "score": timeliness_score,
                "weight": "10%",
                "date_columns": date_cols,
                "details": "数据新鲜度",
            },
        },
        "recommendations": recommendations,
    }


# ======================================================================
# 10. data_export_pipeline — 数据导出管线
# ======================================================================

def data_export_pipeline(data, formats, compress=False):
    """数据导出管线：支持多种格式输出和压缩。

    支持的格式:
      - 'csv': 标准CSV
      - 'json': JSON格式
      - 'tsv': Tab分隔
      - 'excel_xml': Excel 2003 XML格式（SpreadsheetML）
      - 'jsonl': JSON Lines格式

    压缩:
      - 使用GZIP压缩输出

    Args:
        data (list of dict): 数据集
        formats (list): 输出格式列表
        compress (bool): 是否GZIP压缩

    Returns:
        dict: {
            "formats": {format_name: {"content": str/bytes, "size": int, "compressed": bool}},
            "total_size": int,
            "row_count": int,
            "column_count": int,
        }
    """
    if not data:
        return {"formats": {}, "total_size": 0, "row_count": 0, "column_count": 0}

    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    all_columns = sorted(all_columns)

    results = {}
    total_size = 0

    for fmt in formats:
        if fmt == 'csv':
            content = _export_csv(data, all_columns, delimiter=',')
        elif fmt == 'tsv':
            content = _export_csv(data, all_columns, delimiter='\t')
        elif fmt == 'json':
            content = _export_json(data)
        elif fmt == 'jsonl':
            content = _export_jsonl(data)
        elif fmt == 'excel_xml':
            content = _export_excel_xml(data, all_columns)
        else:
            continue

        content_bytes = content.encode('utf-8') if isinstance(content, str) else content

        if compress:
            compressed = gzip.compress(content_bytes)
            results[fmt] = {
                "content": content,
                "compressed_content": compressed,
                "size": len(compressed),
                "original_size": len(content_bytes),
                "compressed": True,
                "compression_ratio": round(len(compressed) / len(content_bytes) * 100, 2) if content_bytes else 0,
            }
            total_size += len(compressed)
        else:
            results[fmt] = {
                "content": content,
                "size": len(content_bytes),
                "compressed": False,
            }
            total_size += len(content_bytes)

    return {
        "formats": results,
        "total_size": total_size,
        "row_count": len(data),
        "column_count": len(all_columns),
    }


def _export_csv(data, columns, delimiter=','):
    """导出CSV/TSV格式。"""
    output = StringIO()
    writer = csv.writer(output, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)
    for row in data:
        writer.writerow([_format_export_value(row.get(col)) for col in columns])
    return output.getvalue()


def _export_json(data):
    """导出JSON格式。"""
    def _default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(data, ensure_ascii=False, indent=2, default=_default)


def _export_jsonl(data):
    """导出JSON Lines格式。"""
    lines = []
    for row in data:
        def _default(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)
        lines.append(json.dumps(row, ensure_ascii=False, default=_default))
    return "\n".join(lines)


def _export_excel_xml(data, columns):
    """导出Excel 2003 XML格式（SpreadsheetML）。"""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
        '<Worksheet ss:Name="Sheet1">',
        '<Table>',
    ]

    # 表头
    lines.append('<Row>')
    for col in columns:
        lines.append(f'<Cell><Data ss:Type="String">{_xml_escape(col)}</Data></Cell>')
    lines.append('</Row>')

    # 数据行
    for row in data:
        lines.append('<Row>')
        for col in columns:
            val = row.get(col)
            if val is None or val == "":
                lines.append('<Cell><Data ss:Type="String"></Data></Cell>')
            elif isinstance(val, bool):
                lines.append(f'<Cell><Data ss:Type="Boolean">{"1" if val else "0"}</Data></Cell>')
            elif isinstance(val, int):
                lines.append(f'<Cell><Data ss:Type="Number">{val}</Data></Cell>')
            elif isinstance(val, float):
                lines.append(f'<Cell><Data ss:Type="Number">{val}</Data></Cell>')
            else:
                lines.append(f'<Cell><Data ss:Type="String">{_xml_escape(str(val))}</Data></Cell>')
        lines.append('</Row>')

    lines.extend(['</Table>', '</Worksheet>', '</Workbook>'])
    return "\n".join(lines)


def _xml_escape(s):
    """XML特殊字符转义。"""
    s = str(s)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


def _format_export_value(val):
    """格式化导出值。"""
    if val is None:
        return ""
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, datetime):
        return val.isoformat()
    else:
        return str(val)


# ======================================================================
# 主程序测试
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("data-cleaner 测试")
    print("=" * 60)

    # 测试数据
    test_data = [
        {"name": "Alice", "age": 30, "salary": 50000, "dept": "IT", "join_date": "2022-01-15"},
        {"name": "Bob", "age": 25, "salary": 45000, "dept": "HR", "join_date": "2023-03-20"},
        {"name": "Charlie", "age": 35, "salary": 60000, "dept": "IT", "join_date": "2021-07-01"},
        {"name": "David", "age": 28, "salary": None, "dept": "Finance", "join_date": "2023-06-10"},
        {"name": "Eve", "age": 200, "salary": 55000, "dept": "IT", "join_date": "2022-11-30"},
        {"name": "Frank", "age": 32, "salary": 48000, "dept": "HR", "join_date": "2023-01-05"},
        {"name": "Grace", "age": 29, "salary": 52000, "dept": "Finance", "join_date": "2022-09-15"},
        {"name": "Alice", "age": 30, "salary": 50000, "dept": "IT", "join_date": "2022-01-15"},  # 重复
    ]

    # 1. 数据画像
    print("\n--- 1. data_profiling ---")
    profile = data_profiling(test_data)
    print(f"行数: {profile['row_count']}, 列数: {profile['column_count']}")
    for col, info in profile["columns"].items():
        print(f"  {col}: type={info['type']}, unique={info['unique']}, missing={info['missing']}")
        if info["stats"]:
            print(f"    stats: mean={info['stats'].get('mean')}, std={info['stats'].get('std')}")

    # 2. Z-score异常检测
    print("\n--- 2. outlier_detector_zscore ---")
    values = [row["age"] for row in test_data if row.get("age") is not None]
    z_result = outlier_detector_zscore(values, threshold=2.5)
    print(f"总数: {z_result['total']}, 异常: {z_result['outliers']}")
    print(f"中位数: {z_result['median']}, MAD: {z_result['mad']}")
    for d in z_result["details"]:
        if d["is_outlier"]:
            print(f"  异常: index={d['index']}, value={d['value']}, z={d['z_score']}")

    # 3. Isolation Forest
    print("\n--- 3. outlier_detector_isolation_forest ---")
    if_result = outlier_detector_isolation_forest(test_data, n_trees=20, sample_size=8)
    print(f"总数: {if_result['total']}, 异常: {if_result['outliers']}")
    for s in if_result["scores"]:
        if s["is_outlier"]:
            print(f"  异常: index={s['index']}, score={s['score']}")

    # 4. 缺失值填充
    print("\n--- 4. missing_value_imputer ---")
    knn_result = missing_value_imputer(test_data, strategy='knn', k=3)
    print(f"策略: {knn_result['strategy']}")
    print(f"缺失: {knn_result['total_missing']}, 填充: {knn_result['imputed']}")
    for col, detail in knn_result["column_details"].items():
        if detail["missing_before"] > 0:
            print(f"  {col}: 缺失={detail['missing_before']}, 填充={detail['imputed']}")

    # 5. 重复检测
    print("\n--- 5. duplicate_detector ---")
    dup_result = duplicate_detector(test_data, method='fuzzy')
    print(f"总记录: {dup_result['total_records']}, 重复对: {dup_result['duplicate_pairs']}")
    print(f"唯一记录: {dup_result['unique_records']}")
    for d in dup_result["duplicates"]:
        print(f"  重复: {d['record1_index']} <-> {d['record2_index']}, 相似度={d['similarity']}")

    # 6. 数据标准化
    print("\n--- 6. data_normalizer ---")
    salaries = [row["salary"] for row in test_data if row.get("salary") is not None]
    for method in ['minmax', 'zscore', 'robust']:
        norm_result = data_normalizer(salaries, method=method)
        print(f"  {method}: params={norm_result['params']}")
        print(f"    前3个值: {norm_result['normalized_data'][:3]}")

    # 7. 相关性分析
    print("\n--- 7. correlation_analyzer ---")
    corr_result = correlation_analyzer(test_data, method='pearson')
    print(f"方法: {corr_result['method']}")
    print(f"数值列: {corr_result['columns']}")
    for pair in corr_result["significant_pairs"]:
        print(f"  {pair['col1']} vs {pair['col2']}: r={pair['r']}, p={pair['p_value']} ({pair['strength']})")

    # 8. 类型转换
    print("\n--- 8. data_type_converter ---")
    type_result = data_type_converter(test_data)
    print(f"推断类型: {type_result['inferred_types']}")
    for col, stats in type_result["conversion_stats"].items():
        print(f"  {col}: 转换率={stats['rate']}%")

    # 9. 质量评分
    print("\n--- 9. data_quality_scorer ---")
    quality = data_quality_scorer(test_data)
    print(f"综合评分: {quality['overall_score']}/100 ({quality['grade']})")
    for dim, info in quality["dimensions"].items():
        print(f"  {dim}: {info['score']} ({info['weight']})")
    for rec in quality["recommendations"]:
        print(f"  建议: {rec}")

    # 10. 导出管线
    print("\n--- 10. data_export_pipeline ---")
    export_result = data_export_pipeline(
        test_data,
        formats=['csv', 'json', 'tsv', 'jsonl'],
        compress=True
    )
    print(f"行数: {export_result['row_count']}, 列数: {export_result['column_count']}")
    for fmt, info in export_result["formats"].items():
        print(f"  {fmt}: 压缩后大小={info['size']}B, 原始大小={info.get('original_size', info['size'])}B")
        if "compression_ratio" in info:
            print(f"    压缩率: {info['compression_ratio']}%")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
