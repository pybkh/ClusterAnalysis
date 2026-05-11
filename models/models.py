"""
数据模型层 (Model)
负责数据的加载、清洗、存储和管理
纯 Python 逻辑，不依赖 PySide6
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .utils import (
    COLUMN_NAME_MAP, normalize_column_name, is_feature_column,
    find_header_row, detect_multiple_data_blocks,
    normalize_variety_name,
    FEATURE_COLUMNS, NON_FEATURE_COLUMNS
)


@dataclass
class LoadResult:
    """文件加载结果"""
    success: bool
    raw_df: Optional[pd.DataFrame] = None
    error_msg: Optional[str] = None
    stats: Optional[Dict] = None  # 统计信息：数据块数、有效行数、丢弃行数


class DataManager:
    """
    数据管理器类

    封装数据加载、清洗、存储和管理功能：
    - 读取 CSV/Excel 文件
    - 智能表头搜索和列名标准化
    - 多数据块检测和拆分
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
        self._load_stats: Dict = {}  # 加载统计信息

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

    @property
    def load_stats(self) -> Dict:
        """获取加载统计信息"""
        return self._load_stats.copy()

    def get_connection(self) -> Optional[sqlite3.Connection]:
        """
        获取数据库连接

        Returns:
            sqlite3.Connection 或 None
        """
        return self._conn

    # ==================== 文件读取功能 ====================

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
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.csv':
                # 处理 CSV 编码：先尝试 utf-8，失败再尝试 gbk
                try:
                    return pd.read_csv(file_path, encoding='utf-8', header=None)
                except (UnicodeDecodeError, UnicodeError):
                    return pd.read_csv(file_path, encoding='gbk', header=None)

            elif ext in ['.xlsx', '.xls']:
                # 当未指定 sheet_name 时，读取第一个工作表
                if sheet_name is None:
                    with pd.ExcelFile(file_path) as xl:
                        sheet_name = xl.sheet_names[0]
                return pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            else:
                raise ValueError(f"不支持的文件格式: {ext}")

        except Exception as e:
            raise Exception(f"读取文件失败: {str(e)}")

    def read_multiple_sheets(self, file_path: str, sheet_names: List[str]) -> pd.DataFrame:
        """
        读取多个 Excel 工作表并合并

        Args:
            file_path: Excel 文件路径
            sheet_names: 要合并的工作表名称列表

        Returns:
            pd.DataFrame: 合并后的 DataFrame，包含 source_sheet 列
        """
        dfs = []
        for sheet_name in sheet_names:
            df = self.read_file(file_path, sheet_name)
            if df is not None:
                df['source_sheet'] = sheet_name
                dfs.append(df)

        if not dfs:
            raise ValueError("未能读取任何工作表数据")

        return pd.concat(dfs, ignore_index=True)

    def get_excel_sheets(self, file_path: str) -> List[str]:
        """
        获取 Excel 文件的所有工作表名称

        Args:
            file_path: Excel 文件路径

        Returns:
            List[str]: 工作表名称列表
        """
        try:
            with pd.ExcelFile(file_path) as xl:
                return xl.sheet_names
        except Exception as e:
            raise Exception(f"获取 Excel 工作表失败: {str(e)}")

    # ==================== 数据加载功能 ====================

    def load_file(self, file_path: str, sheet_name: Optional[str] = None,
                  merge_all_sheets: bool = False) -> LoadResult:
        """
        加载文件并导入内存数据库

        流程：
        1. 读取原始数据（支持多sheet合并）
        2. 智能查找表头行（优先搜索"物体编号"）
        3. 检测多个数据块并拆分
        4. 标准化列名（中文→英文）
        5. 清洗数据（数值转换）
        6. 写入 sqlite3 内存数据库

        Args:
            file_path: 文件路径（CSV 或 Excel）
            sheet_name: Excel 工作表名（仅对 Excel 文件有效）
            merge_all_sheets: 是否合并所有工作表

        Returns:
            LoadResult: 加载结果，包含成功/失败状态和统计信息
        """
        try:
            # 1. 读取原始数据
            ext = os.path.splitext(file_path)[1].lower()
            source_sheet = sheet_name

            if merge_all_sheets and ext in ['.xlsx', '.xls']:
                # 合并所有工作表
                sheets = self.get_excel_sheets(file_path)
                raw_df = self.read_multiple_sheets(file_path, sheets)
                source_sheet = 'merged'
            elif ext in ['.xlsx', '.xls']:
                raw_df = self.read_file(file_path, sheet_name)
                source_sheet = sheet_name or 'Sheet1'
            else:
                raw_df = self.read_file(file_path)
                source_sheet = 'CSV'

            if raw_df is None:
                return LoadResult(success=False, error_msg="无法读取文件，请检查文件格式和路径")

            if raw_df.empty:
                return LoadResult(success=False, error_msg="文件为空，没有数据")

            # 2. 智能查找表头行
            header_row = find_header_row(raw_df)

            if header_row == -1:
                # 未找到表头，返回让用户手动选择
                return LoadResult(success=False, raw_df=raw_df)

            # 3. 检测多个数据块
            blocks = detect_multiple_data_blocks(raw_df, header_row)

            # 4. 处理数据
            stats = self._process_data_blocks(raw_df, blocks, source_sheet)
            self._load_stats = stats

            return LoadResult(success=True, stats=stats)

        except Exception as e:
            return LoadResult(success=False, error_msg=str(e))

    def load_with_manual_header(self, raw_df: pd.DataFrame, header_row: int,
                                 source_sheet: str = 'manual') -> LoadResult:
        """
        使用手动指定的表头行加载数据

        Args:
            raw_df: 原始 DataFrame
            header_row: 手动指定的表头行索引
            source_sheet: 来源工作表名称

        Returns:
            LoadResult: 加载结果
        """
        try:
            blocks = [(header_row, None)]
            stats = self._process_data_blocks(raw_df, blocks, source_sheet)
            self._load_stats = stats
            return LoadResult(success=True, stats=stats)
        except Exception as e:
            return LoadResult(success=False, error_msg=str(e))

    def _process_data_blocks(self, raw_df: pd.DataFrame, blocks: List[tuple],
                              source_sheet: str) -> Dict:
        """
        处理多个数据块

        Args:
            raw_df: 原始 DataFrame
            blocks: 数据块列表 [(header_row, variety_name), ...]
            source_sheet: 来源工作表名称

        Returns:
            Dict: 统计信息
        """
        all_data = []
        stats = {
            'block_count': len(blocks),
            'blocks': [],
            'total_rows': 0,
            'valid_rows': 0,
            'discarded_rows': 0
        }
        seen_variety_names = set()  # 用于标准化时避免重复

        for i, (header_row, variety_name) in enumerate(blocks):
            # 确定数据区结束位置（下一个表头行或文件末尾）
            if i + 1 < len(blocks):
                end_row = blocks[i + 1][0]
            else:
                end_row = len(raw_df)

            # 提取表头和数据
            headers = raw_df.iloc[header_row].values
            data_df = raw_df.iloc[header_row + 1:end_row].copy()
            data_df.columns = headers

            # 标准化列名
            data_df.columns = [normalize_column_name(str(col)) for col in data_df.columns]

            # 品种名称标准化：未找到时降级使用 sheet 名
            if not variety_name or variety_name == '未知品种':
                variety_name = source_sheet or f'未知品种_{i}'
            variety_name = normalize_variety_name(variety_name, seen_variety_names, i)
            seen_variety_names.add(variety_name)

            # 添加来源信息和品种名称
            data_df['source_sheet'] = source_sheet
            data_df['variety_name'] = variety_name

            # 清洗数据
            original_count = len(data_df)
            self._clean_data(data_df)
            valid_count = len(data_df)
            discarded = original_count - valid_count

            # 统计信息
            block_stats = {
                'block_index': i + 1,
                'variety_name': variety_name,
                'total_rows': original_count,
                'valid_rows': valid_count,
                'discarded_rows': discarded
            }
            stats['blocks'].append(block_stats)
            stats['total_rows'] += original_count
            stats['valid_rows'] += valid_count
            stats['discarded_rows'] += discarded

            if not data_df.empty:
                all_data.append(data_df)

        if not all_data:
            raise ValueError("所有数据块均为空或无效")

        # 合并所有数据块
        merged_df = pd.concat(all_data, ignore_index=True)

        # 验证数据
        self._validate_data(merged_df)

        # 写入数据库
        self._write_to_database(merged_df)

        # 更新列信息
        self._update_column_info(merged_df)

        return stats

    def _clean_data(self, df: pd.DataFrame) -> None:
        """
        清洗数据：
        - 删除全为 NaN 的行
        - 数值列强制转为 float
        - 至少包含 2 个非空数值单元格的行才保留

        Args:
            df: 要清洗的 DataFrame（原地修改）
        """
        # 1. 删除全为 NaN 的行
        df.dropna(how='all', inplace=True)

        # 2. 数值列强制转为 float
        numeric_cols = []
        for col in df.columns:
            if col in NON_FEATURE_COLUMNS or col in ['source_sheet', 'source_group']:
                continue
            # 尝试转换为数值类型，无法转换的设为 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)

        # 3. 过滤无效行：至少包含 2 个非空数值单元格
        if numeric_cols:
            non_null_counts = df[numeric_cols].notna().sum(axis=1)
            df.drop(df[non_null_counts < 2].index, inplace=True)

        # 4. 重置索引
        df.reset_index(drop=True, inplace=True)

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

        增加固定列：row_id（自增主键）、original_db_index

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

        # 添加 row_id 自增主键
        # 根据列的实际类型确定 SQLite 类型
        col_types = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'int32']:
                col_types.append(f'"{col}" INTEGER')
            elif df[col].dtype in ['float64', 'float32']:
                col_types.append(f'"{col}" REAL')
            else:
                col_types.append(f'"{col}" TEXT')

        self._conn.execute(f"""
            CREATE TABLE {self._table_name}_new (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                {', '.join(col_types)}
            )
        """)
        self._conn.execute(f"""
            INSERT INTO {self._table_name}_new ({', '.join([f'"{col}"' for col in df.columns])})
            SELECT {', '.join([f'"{col}"' for col in df.columns])} FROM {self._table_name}
        """)
        self._conn.execute(f"DROP TABLE {self._table_name}")
        self._conn.execute(f"ALTER TABLE {self._table_name}_new RENAME TO {self._table_name}")
        self._conn.commit()

        # 更新行数
        self._row_count = len(df)

    def _update_column_info(self, df: pd.DataFrame) -> None:
        """
        更新列信息

        Args:
            df: 数据 DataFrame
        """
        self._columns = list(df.columns)
        # 确保 row_id 在列中
        if 'row_id' not in self._columns:
            self._columns.insert(0, 'row_id')
        self._feature_columns = [c for c in self._columns if is_feature_column(c)]

    # ==================== 数据查询功能 ====================

    def get_row_count(self) -> int:
        """获取数据库中的数据行数"""
        if not self._conn:
            return 0
        try:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            return cursor.fetchone()[0]
        except:
            return 0

    def get_all_data(self) -> Optional[pd.DataFrame]:
        """获取数据库中的所有数据"""
        if not self._conn:
            return None
        try:
            return pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn)
        except:
            return None

    def get_source_groups(self) -> List[str]:
        """
        获取所有品种（variety_name）列表

        Returns:
            List[str]: 品种名称列表
        """
        if not self._conn:
            return []

        try:
            # 检查 variety_name 列是否存在
            cursor = self._conn.execute(f"PRAGMA table_info({self._table_name})")
            columns = [row[1] for row in cursor.fetchall()]

            if 'variety_name' not in columns:
                return ['未知品种']

            cursor = self._conn.execute(
                f"SELECT DISTINCT variety_name FROM {self._table_name} WHERE variety_name IS NOT NULL"
            )
            groups = [row[0] for row in cursor.fetchall()]
            return groups if groups else ['未知品种']
        except:
            return ['未知品种']

    def get_variety_sample_counts(self) -> Dict[str, int]:
        """
        获取每个品种的样本数量

        Returns:
            Dict[str, int]: {品种名称: 样本数}
        """
        if not self._conn:
            return {}
        try:
            df = pd.read_sql_query(
                f"SELECT variety_name, COUNT(*) as cnt FROM {self._table_name} "
                f"WHERE variety_name IS NOT NULL GROUP BY variety_name",
                self._conn
            )
            return dict(zip(df['variety_name'], df['cnt']))
        except:
            return {}

    def get_filtered_data(self, variety_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取按品种过滤的数据（用于表格显示）

        Args:
            variety_name: 品种名称，None 表示获取所有数据

        Returns:
            pd.DataFrame 或 None
        """
        if not self._conn:
            return None
        try:
            if variety_name:
                query = f"SELECT * FROM {self._table_name} WHERE variety_name = '{variety_name}'"
            else:
                query = f"SELECT * FROM {self._table_name}"
            df = pd.read_sql_query(query, self._conn)
            return df
        except:
            return None

    def get_data_by_group(self, selected_features: List[str],
                          variety_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        从数据库获取指定品种用于聚类的数据

        Args:
            selected_features: 用户选择的特征列列表
            variety_name: 品种名称，None 表示获取所有数据

        Returns:
            pd.DataFrame 或 None
        """
        if not self._conn:
            return None

        try:
            cols = selected_features + ['original_db_index', 'variety_name']
            query = f"SELECT {', '.join(cols)} FROM {self._table_name}"

            if variety_name and variety_name != '所有品种':
                query += f" WHERE variety_name = '{variety_name}'"

            df = pd.read_sql_query(query, self._conn)
            return df
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None

    def get_data_for_clustering(self, selected_features: List[str]) -> Optional[pd.DataFrame]:
        """
        从数据库获取用于聚类的数据（所有品种）

        Args:
            selected_features: 用户选择的特征列列表

        Returns:
            pd.DataFrame 或 None
        """
        return self.get_data_by_group(selected_features, variety_name=None)

    # ==================== 数据更新功能 ====================

    def update_cluster_ids(self, cluster_results: pd.DataFrame) -> None:
        """
        将聚类结果更新回数据库

        Args:
            cluster_results: 包含 original_db_index 和 cluster_id 的 DataFrame
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

    def update_cluster_ids_by_group(self, group_results: Dict[str, pd.DataFrame]) -> None:
        """
        将多品种聚类结果更新回数据库

        Args:
            group_results: {品种名: DataFrame(original_db_index, cluster_id)}
        """
        if not self._conn:
            return

        try:
            # 先清除所有聚类结果
            self._conn.execute(f"UPDATE {self._table_name} SET cluster_id = NULL")
            self._conn.commit()

            # 按品种更新
            for group_name, result_df in group_results.items():
                if result_df is not None and not result_df.empty:
                    for _, row in result_df.iterrows():
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

    def rename_variety(self, old_name: str, new_name: str) -> bool:
        """
        重命名品种名称（用于手动修正）

        Args:
            old_name: 原品种名称
            new_name: 新品种名称

        Returns:
            bool: 是否成功
        """
        if not self._conn:
            return False
        try:
            self._conn.execute(
                f"UPDATE {self._table_name} SET variety_name = ? WHERE variety_name = ?",
                (new_name, old_name)
            )
            self._conn.commit()
            # 更新内存中的列信息
            return True
        except Exception as e:
            raise Exception(f"重命名品种失败: {str(e)}")

    # ==================== 数据导出功能 ====================

    def export_to_excel(self, file_path: str, use_chinese_headers: bool = False,
                        split_by_group: bool = False) -> bool:
        """
        导出数据到 Excel 文件

        Args:
            file_path: 保存路径
            use_chinese_headers: 是否使用中文表头
            split_by_group: 是否按品种拆分到不同 sheet

        Returns:
            bool: 是否成功
        """
        if not self._conn:
            return False

        try:
            df = pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn)

            if use_chinese_headers:
                from .utils import get_display_name
                df.columns = [get_display_name(col) for col in df.columns]

            if split_by_group and 'variety_name' in df.columns:
                # 按品种拆分到不同 sheet
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 使用中文或英文列名获取品种列
                    variety_col = '品种名称' if use_chinese_headers else 'variety_name'
                    groups = df[variety_col].unique()
                    for group in groups:
                        group_df = df[df[variety_col] == group].copy()
                        # cluster_id 显示为数字（0,1,2...），同一品种内连续
                        cluster_col = '聚类编号' if use_chinese_headers else 'cluster_id'
                        if cluster_col in group_df.columns:
                            # 重新编号，从0开始
                            unique_clusters = sorted(group_df[cluster_col].dropna().unique())
                            cluster_map = {old: new for new, old in enumerate(unique_clusters)}
                            group_df[cluster_col] = group_df[cluster_col].map(cluster_map)
                        # Sheet 名称限制为 31 字符
                        sheet_name = str(group)[:31]
                        group_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
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
