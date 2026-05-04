"""
数据模型层 (Model)
负责数据的加载、清洗、存储和管理
纯 Python 逻辑，不依赖 PySide6
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from .utils import (
    COLUMN_NAME_MAP, normalize_column_name, is_feature_column,
    find_header_row, FEATURE_COLUMNS, NON_FEATURE_COLUMNS
)


class DataManager:
    """
    数据管理器类

    封装数据加载、清洗、存储和管理功能：
    - 读取 CSV/Excel 文件（整合自 data_loader.py）
    - 智能表头搜索和列名标准化
    - 数据清洗（数值转换、空值处理）
    - 将结果写入 sqlite3 内存数据库

    注意：此类不依赖 PySide6，是纯 Python 逻辑层
    """

    def __init__(self):
        """初始化数据管理器"""
        self._conn: Optional[sqlite3.Connection] = None
        self._table_name = 'phenotype_data'
        self._columns: List[str] = []
        self._feature_columns: List[str] = []
        self._row_count: int = 0

    @property
    def table_name(self) -> str:
        """获取数据库表名"""
        return self._table_name

    @property
    def columns(self) -> List[str]:
        """获取所有列名列表"""
        return self._columns.copy()

    @property
    def feature_columns(self) -> List[str]:
        """获取特征列名列表"""
        return self._feature_columns.copy()

    @property
    def row_count(self) -> int:
        """获取数据行数"""
        return self._row_count

    def get_connection(self) -> Optional[sqlite3.Connection]:
        """
        获取数据库连接

        Returns:
            sqlite3.Connection 或 None
        """
        return self._conn

    # ==================== 文件读取功能（整合自 data_loader.py）====================

    def read_file(self, file_path: str, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        读取 CSV/Excel 文件，返回原始 DataFrame

        Args:
            file_path: 文件路径
            sheet_name: Excel 工作表名（仅对 Excel 文件有效）

        Returns:
            pd.DataFrame 或 None

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            Exception: 读取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.csv':
                # 处理 CSV 编码：先尝试 utf-8，失败再尝试 gbk
                try:
                    return pd.read_csv(file_path, encoding='utf-8')
                except (UnicodeDecodeError, UnicodeError):
                    return pd.read_csv(file_path, encoding='gbk')

            elif ext in ['.xlsx', '.xls']:
                # 当未指定 sheet_name 时，读取第一个工作表
                # 如果不指定，pd.read_excel 会返回字典而非 DataFrame
                if sheet_name is None:
                    # 获取第一个 sheet 名称
                    with pd.ExcelFile(file_path) as xl:
                        sheet_name = xl.sheet_names[0]
                return pd.read_excel(file_path, sheet_name=sheet_name)

            else:
                raise ValueError(f"不支持的文件格式: {ext}")

        except Exception as e:
            raise Exception(f"读取文件失败: {str(e)}")

    def get_excel_sheets(self, file_path: str) -> List[str]:
        """
        获取 Excel 文件的所有工作表名称

        Args:
            file_path: Excel 文件路径

        Returns:
            List[str]: 工作表名称列表

        Raises:
            Exception: 获取失败
        """
        try:
            with pd.ExcelFile(file_path) as xl:
                return xl.sheet_names
        except Exception as e:
            raise Exception(f"获取 Excel 工作表失败: {str(e)}")

    # ==================== 数据加载功能 ====================

    def load_file(self, file_path: str, sheet_name: Optional[str] = None) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        加载文件并导入内存数据库

        流程：
        1. 读取原始数据
        2. 智能查找表头行
        3. 标准化列名（中文→英文）
        4. 清洗数据（数值转换）
        5. 写入 sqlite3 内存数据库

        Args:
            file_path: 文件路径（CSV 或 Excel）
            sheet_name: Excel 工作表名（仅对 Excel 文件有效）

        Returns:
            Tuple[bool, Optional[pd.DataFrame]]:
                - (True, None): 加载成功
                - (False, raw_df): 未找到表头，返回原始数据让用户手动选择

        Raises:
            ValueError: 文件为空或数据不足
            Exception: 其他加载错误
        """
        try:
            # 1. 读取原始数据
            raw_df = self.read_file(file_path, sheet_name)
            if raw_df is None:
                raise ValueError("无法读取文件，请检查文件格式和路径")

            # 检查是否为空文件
            if raw_df.empty:
                raise ValueError("文件为空，没有数据")

            # 2. 智能查找表头行
            header_row = find_header_row(raw_df)

            if header_row == -1:
                # 未找到表头，返回让用户手动选择
                return False, raw_df

            # 3. 处理数据
            self._process_data(raw_df, header_row)
            return True, None

        except Exception as e:
            raise e

    def load_with_manual_header(self, raw_df: pd.DataFrame, header_row: int) -> None:
        """
        使用手动指定的表头行加载数据

        当自动表头识别失败时，由用户手动指定表头行号

        Args:
            raw_df: 原始 DataFrame
            header_row: 手动指定的表头行索引

        Raises:
            ValueError: 数据不足
        """
        try:
            self._process_data(raw_df, header_row)
        except Exception as e:
            raise e

    def _process_data(self, raw_df: pd.DataFrame, header_row: int) -> None:
        """
        处理数据：提取表头、清洗、写入数据库

        Args:
            raw_df: 原始 DataFrame
            header_row: 表头行索引
        """
        # 提取表头和数据
        headers = raw_df.iloc[header_row].values
        data_df = raw_df.iloc[header_row + 1:].copy()
        data_df.columns = headers

        # 标准化列名
        data_df.columns = [normalize_column_name(str(col)) for col in data_df.columns]

        # 清洗数据：删除空行、数值列强制转为 float
        self._clean_data(data_df)

        # 检查有效数据量
        self._validate_data(data_df)

        # 写入内存数据库
        self._write_to_database(data_df)

        # 更新列信息
        self._update_column_info(data_df)

    def _clean_data(self, df: pd.DataFrame) -> None:
        """
        清洗数据：删除空行、将数值列强制转为 float

        Args:
            df: 要清洗的 DataFrame（原地修改）
        """
        # 1. 删除全为 NaN 的行
        df.dropna(how='all', inplace=True)

        # 2. 删除关键特征列全为空的行（至少需要一个特征有值）
        feature_cols = [c for c in df.columns if is_feature_column(c)]
        if feature_cols:
            df.dropna(subset=feature_cols, how='all', inplace=True)

        # 3. 重置索引
        df.reset_index(drop=True, inplace=True)

        # 4. 数值列强制转为 float
        for col in df.columns:
            # 跳过非数值列
            if col in NON_FEATURE_COLUMNS:
                continue
            # 尝试转换为数值类型，无法转换的设为 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        验证数据有效性

        Args:
            df: 要验证的 DataFrame

        Raises:
            ValueError: 有效数据量不足
        """
        numeric_cols = [c for c in df.columns if is_feature_column(c)]
        if numeric_cols:
            valid_rows = df[numeric_cols].dropna(how='all').shape[0]
            if valid_rows < 5:
                raise ValueError(f"有效数据量不足（仅 {valid_rows} 行），至少需要 5 行数据")

    def _write_to_database(self, df: pd.DataFrame) -> None:
        """
        将 DataFrame 写入 sqlite3 内存数据库

        Args:
            df: 要写入的 DataFrame
        """
        # 关闭旧连接
        if self._conn:
            self._conn.close()

        # 创建新的内存数据库连接
        self._conn = sqlite3.connect(':memory:')

        # 添加原始行索引列（用于后续图表与表格的交互）
        df = df.copy()
        df['original_db_index'] = range(len(df))

        # 使用 pandas to_sql 写入数据库
        df.to_sql(self._table_name, self._conn, index=False, if_exists='replace')

        # 更新行数
        self._row_count = len(df)

    def _update_column_info(self, df: pd.DataFrame) -> None:
        """
        更新列信息

        Args:
            df: 数据 DataFrame
        """
        self._columns = list(df.columns)
        self._feature_columns = [c for c in self._columns if is_feature_column(c)]

    # ==================== 数据查询功能 ====================

    def get_row_count(self) -> int:
        """
        获取数据库中的数据行数

        Returns:
            int: 行数
        """
        if not self._conn:
            return 0
        try:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            return cursor.fetchone()[0]
        except:
            return 0

    def get_all_data(self) -> Optional[pd.DataFrame]:
        """
        获取数据库中的所有数据

        Returns:
            pd.DataFrame 或 None
        """
        if not self._conn:
            return None
        try:
            return pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn)
        except:
            return None

    def get_data_for_clustering(self, selected_features: List[str]) -> Optional[pd.DataFrame]:
        """
        从数据库获取用于聚类的数据

        只读取用户选中的特征列和原始索引列，减少内存占用

        Args:
            selected_features: 用户选择的特征列列表

        Returns:
            pd.DataFrame 或 None
        """
        if not self._conn:
            return None

        try:
            cols = selected_features + ['original_db_index']
            query = f"SELECT {', '.join(cols)} FROM {self._table_name}"
            df = pd.read_sql_query(query, self._conn)
            return df
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None

    # ==================== 数据更新功能 ====================

    def update_cluster_ids(self, cluster_results: pd.DataFrame) -> None:
        """
        将聚类结果更新回数据库

        Args:
            cluster_results: 包含 original_db_index 和 cluster_id 的 DataFrame

        Raises:
            Exception: 更新失败时抛出异常
        """
        if not self._conn:
            return

        try:
            # 添加 cluster_id 列（如果不存在）
            try:
                self._conn.execute(f"ALTER TABLE {self._table_name} ADD COLUMN cluster_id INTEGER")
            except:
                pass  # 列已存在

            # 更新聚类结果
            for _, row in cluster_results.iterrows():
                self._conn.execute(
                    f"UPDATE {self._table_name} SET cluster_id = ? WHERE original_db_index = ?",
                    (int(row['cluster_id']), int(row['original_db_index']))
                )

            self._conn.commit()

            # 更新列信息
            if 'cluster_id' not in self._columns:
                self._columns.append('cluster_id')

        except Exception as e:
            raise Exception(f"更新聚类结果失败: {str(e)}")

    # ==================== 数据导出功能 ====================

    def export_to_excel(self, file_path: str) -> bool:
        """
        导出数据到 Excel 文件

        Args:
            file_path: 保存路径

        Returns:
            bool: 是否成功

        Raises:
            Exception: 导出失败时抛出异常
        """
        if not self._conn:
            return False

        try:
            df = pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn)
            df.to_excel(file_path, index=False)
            return True
        except Exception as e:
            raise Exception(f"导出失败: {str(e)}")

    # ==================== 资源清理 ====================

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
