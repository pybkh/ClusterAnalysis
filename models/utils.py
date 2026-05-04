"""
工具函数模块
提供中文列名映射、数据清洗等通用功能
"""

import pandas as pd

# 中文列名到英文的映射字典（用于数据导入时标准化列名）
COLUMN_NAME_MAP = {
    '粒宽': 'width',
    '粒高': 'height',
    '周长': 'perimeter',
    '面积': 'area',
    '相似度': 'similarity',
    '相似': 'similarity',
    '物体编号': 'object_id',
    '编号': 'object_id',
}

# 英文列名到中文的反向映射（用于界面显示）
COLUMN_NAME_REVERSE_MAP = {
    'width': '粒宽',
    'height': '粒高',
    'perimeter': '周长',
    'area': '面积',
    'similarity': '相似度',
    'object_id': '物体编号',
    'cluster_id': '聚类编号',
    'original_db_index': '原始索引',
}

# 预定义的特征列（用于聚类分析）
FEATURE_COLUMNS = ['width', 'height', 'perimeter', 'area', 'similarity']

# 特征列的中文显示名
FEATURE_DISPLAY_NAMES = {
    'width': '粒宽',
    'height': '粒高',
    'perimeter': '周长',
    'area': '面积',
    'similarity': '相似度',
}

# 非特征列（元数据列，不参与聚类选择）
NON_FEATURE_COLUMNS = ['object_id', 'cluster_id', 'original_db_index']


def get_display_name(col_name: str) -> str:
    """
    获取列的中文显示名称

    Args:
        col_name: 英文列名

    Returns:
        str: 中文显示名称，如果映射不存在则返回原名
    """
    return COLUMN_NAME_REVERSE_MAP.get(col_name, col_name)


def normalize_column_name(col_name: str) -> str:
    """
    将中文列名标准化为英文列名

    支持精确匹配和部分匹配（如 '面积(mm^2)' -> 'area'）

    Args:
        col_name: 原始列名（可能是中文）

    Returns:
        str: 标准化后的英文列名
    """
    # 1. 精确匹配
    if col_name in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[col_name]

    # 2. 部分匹配（检查列名是否包含映射字典中的关键词）
    for cn_name, en_name in COLUMN_NAME_MAP.items():
        if cn_name in col_name:
            return en_name

    # 3. 无匹配，返回原名
    return col_name


def is_feature_column(col_name: str) -> bool:
    """
    判断列是否为特征列（参与聚类分析）

    Args:
        col_name: 列名

    Returns:
        bool: 是否为特征列
    """
    return col_name in FEATURE_COLUMNS


def find_header_row(df: pd.DataFrame, min_matches: int = 3) -> int:
    """
    智能查找表头行

    遍历所有行，寻找包含至少 min_matches 个预定义关键词的行。
    如果找到多个候选，选择非空单元格比例最高的行。

    Args:
        df: 原始 DataFrame（未处理表头的原始数据）
        min_matches: 最少匹配关键词数量

    Returns:
        int: 表头行索引，如果未找到返回 -1
    """
    keywords = set(COLUMN_NAME_MAP.keys())
    candidates = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])

        # 计算匹配的关键词数量
        match_count = sum(1 for kw in keywords if kw in row_str)

        if match_count >= min_matches:
            # 计算非空单元格比例
            non_null_ratio = row.notna().sum() / len(row)
            candidates.append((idx, match_count, non_null_ratio))

    if not candidates:
        return -1

    # 选择匹配数最多、非空比例最高的候选行
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]
