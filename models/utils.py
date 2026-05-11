"""
工具函数模块
提供中文列名映射、数据清洗等通用功能
"""

import re
from typing import Optional, Set
import pandas as pd

# 中文列名到英文的映射字典（用于数据导入时标准化列名）
COLUMN_NAME_MAP = {
    '粒宽': 'width',
    '粒高': 'height',
    '高(mm)': 'height',
    '高': 'height',
    '周长': 'perimeter',
    '面积(mm^2)': 'area',
    '面积': 'area',
    '相似度': 'similarity',
    '相似': 'similarity',
    '物体编号': 'object_id',
    '编号': 'object_id',
}

# 英文列名到中文的反向映射（用于界面显示和导出）
COLUMN_NAME_REVERSE_MAP = {
    'width': '粒宽',
    'height': '粒高',
    'perimeter': '周长',
    'area': '面积',
    'similarity': '相似度',
    'object_id': '物体编号',
    'cluster_id': '聚类编号',
    'original_db_index': '原始索引',
    'row_id': '行号',
    'source_sheet': '来源工作表',
    'source_group': '来源分组',
    'variety_name': '品种名称',
}

# 非特征列（元数据列，不参与聚类选择）
NON_FEATURE_COLUMNS = ['object_id', 'cluster_id', 'original_db_index', 'row_id',
                       'source_sheet', 'source_group', 'variety_name']

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


# 用于表头搜索的关键词列表（按优先级排序）
HEADER_KEYWORDS = ['物体编号', '粒宽', '粒高', '高(mm)', '周长', '面积(mm^2)', '面积', '相似度']


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
    使用正则忽略列名中的单位后缀和括号

    Args:
        col_name: 原始列名（可能是中文）

    Returns:
        str: 标准化后的英文列名
    """
    # 1. 精确匹配
    if col_name in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[col_name]

    # 2. 正则处理：移除括号和单位后缀，再进行匹配
    # 例如：'面积(mm^2)' -> '面积'，'高(mm)' -> '高'
    cleaned_name = re.sub(r'[\(（].*?[\)）]', '', col_name).strip()
    if cleaned_name in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[cleaned_name]

    # 3. 部分匹配（检查列名是否包含映射字典中的关键词）
    for cn_name, en_name in COLUMN_NAME_MAP.items():
        if cn_name in col_name:
            return en_name

    # 4. 无匹配，返回原名
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


def find_header_row(df: pd.DataFrame) -> int:
    """
    智能查找表头行

    优化策略：
    1. 优先搜索第一个包含"物体编号"的单元格
    2. 若未找到，对前 20 行计算综合评分：评分 = 映射关键词命中数 × 非空单元格比例

    Args:
        df: 原始 DataFrame（未处理表头的原始数据）

    Returns:
        int: 表头行索引，如果未找到返回 -1
    """
    # 策略1：优先搜索"物体编号"
    for idx in range(min(len(df), 50)):  # 最多搜索前50行
        row = df.iloc[idx]
        row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])
        if '物体编号' in row_str:
            return idx

    # 策略2：综合评分算法
    keywords = set(COLUMN_NAME_MAP.keys())
    candidates = []
    search_range = min(len(df), 20)  # 只搜索前20行

    for idx in range(search_range):
        row = df.iloc[idx]
        row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])

        # 计算匹配的关键词数量
        match_count = sum(1 for kw in keywords if kw in row_str)

        if match_count >= 2:  # 至少匹配2个关键词
            # 计算非空单元格比例
            non_null_ratio = row.notna().sum() / len(row)
            # 综合评分 = 关键词命中数 × 非空单元格比例
            score = match_count * non_null_ratio
            candidates.append((idx, score, match_count, non_null_ratio))

    if not candidates:
        return -1

    # 选择综合评分最高的候选行
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_multiple_data_blocks(df: pd.DataFrame, first_header_row: int,
                                 default_variety: str = '未知品种') -> list:
    """
    检测多个数据块并提取品种名称

    某些 Excel 中可能包含多个数据块（即下方再次出现"物体编号"行），
    需向下扫描检测，若发现第二个数据块，自动拆分并分别处理。

    定位表头行（含"物体编号"）后，向上扫描元数据区，寻找包含"品种名称"的单元格。
    提取该单元格所在行、对应数据列的值作为 variety_name。

    Args:
        df: 原始 DataFrame
        first_header_row: 第一个表头行索引
        default_variety: 默认品种名称（未找到时使用）

    Returns:
        list: 数据块列表，每个元素为 (header_row, variety_name)
    """
    # 提取第一个数据块的品种名称
    first_variety = _extract_variety_name(df, first_header_row, default_variety)
    blocks = [(first_header_row, first_variety)]

    current_row = first_header_row + 1

    while current_row < len(df):
        row = df.iloc[current_row]
        row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])

        # 检测是否为新的表头行（包含"物体编号"）
        if '物体编号' in row_str:
            # 提取该数据块的品种名称
            variety_name = _extract_variety_name(df, current_row, default_variety)
            blocks.append((current_row, variety_name))

        current_row += 1

    return blocks


def normalize_variety_name(name: Optional[str], seen_names: set,
                            index: int = 0) -> str:
    """
    标准化品种名称

    - 去除首尾空格、换行符
    - 空值或纯数字 → "未知品种_N"

    Args:
        name: 原始品种名称
        seen_names: 已使用的品种名称集合（用于生成不重复的未知品种名）
        index: 序号（用于生成"未知品种_N"时的基数）

    Returns:
        str: 标准化后的品种名称
    """
    if name is None:
        name = ''
    name = str(name).strip().replace('\n', '').replace('\r', '')

    # 空值或纯数字 → "未知品种_N"
    if not name or (name.replace('.', '').isdigit() and name != ''):
        base = '未知品种'
        n = index
        while f'{base}_{n}' in seen_names:
            n += 1
        return f'{base}_{n}'

    return name


def _extract_variety_name(df: pd.DataFrame, header_row: int,
                          default_variety: str = '未知品种') -> str:
    """
    从表头行上方的元数据区提取品种名称

    向上扫描元数据区，寻找包含"品种名称"或"品种"的单元格。
    提取策略优先级（从高到低）：
    1. 内联格式：单元格值为 "品种名称: 水稻"
    2. 列模式：找到"品种名称"列标题，取其下一行同列的值（需求9.1.2示例）
    3. 同行邻列：同一行其他列的值（兼容旧格式）
    4. 单元格本身提取"品种"后面的文本

    Args:
        df: 原始 DataFrame
        header_row: 表头行索引
        default_variety: 默认品种名称（未找到时使用）

    Returns:
        str: 品种名称
    """
    # 向上扫描元数据区（最多扫描 header_row 行）
    search_start = max(0, header_row - 20)  # 最多向上扫描20行

    for row_idx in range(header_row - 1, search_start - 1, -1):
        row = df.iloc[row_idx]
        row_values = [str(v) for v in row.values if pd.notna(v)]
        row_str = ' '.join(row_values)

        # 寻找包含"品种名称"或"品种"的单元格
        if '品种名称' in row_str or '品种' in row_str:
            # 提取品种名称的值
            # 优先级1: 内联格式 "品种名称: 水稻"
            for val in row_values:
                match = re.search(r'品种(?:名称)?[：:]\s*(.+)', val)
                if match:
                    result = match.group(1).strip()
                    if result and not result.replace('.', '').isdigit():
                        return result

            # 优先级2: 列模式——找到"品种名称"列标题，取下一行同列的值
            # 对应需求9.1.2示例：第1行D列标题为"品种名称"，第2行D列值为"水稻"
            if row_idx + 1 < len(df):
                for col_idx, cell_val in enumerate(row.values):
                    if pd.notna(cell_val) and '品种' in str(cell_val):
                        # 取下一行同列的值
                        next_row_val = df.iloc[row_idx + 1, col_idx]
                        if pd.notna(next_row_val):
                            result = str(next_row_val).strip()
                            if result and '品种' not in result:
                                # 排除纯数字（可能是坐标或序号）
                                if not result.replace('.', '').isdigit():
                                    return result

            # 优先级3: 同行邻列——查找包含"品种"的单元格，取同行其他列的值
            for col_idx, val in enumerate(row.values):
                if pd.notna(val) and '品种' in str(val):
                    for other_idx, other_val in enumerate(row.values):
                        if other_idx != col_idx and pd.notna(other_val):
                            other_str = str(other_val).strip()
                            if other_str and '品种' not in other_str:
                                if not other_str.replace('.', '').isdigit():
                                    return other_str

            # 优先级4: 兜底——从"品种"单元格本身提取
            for val in row_values:
                val_str = val.strip()
                if '品种' in val_str:
                    match = re.search(r'品种(?:名称)?[：:]?\s*(.+)', val_str)
                    if match:
                        result = match.group(1).strip()
                        if result and not result.replace('.', '').isdigit():
                            return result

    # 未找到品种名称，尝试从文件顶部搜索"品种名称"列标题
    # 适用于多数据块共用文件顶部元数据区的场景
    variety_name_col_idx = None
    for row_idx in range(min(10, len(df))):
        row = df.iloc[row_idx]
        for col_idx, cell_val in enumerate(row.values):
            if pd.notna(cell_val) and '品种名称' in str(cell_val):
                variety_name_col_idx = col_idx
                break
        if variety_name_col_idx is not None:
            break

    if variety_name_col_idx is not None:
        # 优先从数据块上方最近的对应列取值（多数据块场景）
        search_top = max(0, header_row - 25)
        for check_row in range(header_row - 1, search_top - 1, -1):
            val = df.iloc[check_row, variety_name_col_idx]
            if pd.notna(val):
                result = str(val).strip()
                if result and '品种' not in result and not result.replace('.', '').isdigit():
                    return result

        # 如果上方未找到，从顶部取第一个值
        for check_row in range(len(df)):
            val = df.iloc[check_row, variety_name_col_idx]
            if pd.notna(val):
                result = str(val).strip()
                if result and '品种' not in result and not result.replace('.', '').isdigit():
                    return result

    # 未找到品种名称，返回默认值
    return default_variety
