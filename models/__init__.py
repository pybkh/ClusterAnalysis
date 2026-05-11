"""
模型层 (Model)
负责数据管理、存储和业务逻辑
纯 Python 逻辑，不依赖 PySide6
"""

from .models import DataManager, LoadResult
from .analyzer import ClusteringParams, ClusteringResult, run_phenotype_clustering, optimize_k_selection
from .utils import (
    COLUMN_NAME_MAP,
    COLUMN_NAME_REVERSE_MAP,
    FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    HEADER_KEYWORDS,
    get_display_name,
    normalize_column_name,
    is_feature_column,
    find_header_row,
    detect_multiple_data_blocks
)

__all__ = [
    'DataManager',
    'LoadResult',
    'ClusteringParams',
    'ClusteringResult',
    'run_phenotype_clustering',
    'optimize_k_selection',
    'COLUMN_NAME_MAP',
    'COLUMN_NAME_REVERSE_MAP',
    'FEATURE_COLUMNS',
    'NON_FEATURE_COLUMNS',
    'HEADER_KEYWORDS',
    'get_display_name',
    'normalize_column_name',
    'is_feature_column',
    'find_header_row',
    'detect_multiple_data_blocks'
]
