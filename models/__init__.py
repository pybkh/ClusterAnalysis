"""
模型层 (Model)
负责数据管理、存储和业务逻辑
纯 Python 逻辑，不依赖 PySide6
"""

from .models import DataManager
from .utils import (
    COLUMN_NAME_MAP,
    COLUMN_NAME_REVERSE_MAP,
    FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    get_display_name,
    normalize_column_name,
    is_feature_column,
    find_header_row
)

__all__ = [
    'DataManager',
    'COLUMN_NAME_MAP',
    'COLUMN_NAME_REVERSE_MAP',
    'FEATURE_COLUMNS',
    'NON_FEATURE_COLUMNS',
    'get_display_name',
    'normalize_column_name',
    'is_feature_column',
    'find_header_row'
]
