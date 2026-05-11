"""
工作线程模块
负责在后台执行耗时的聚类计算任务
使用 QObject + moveToThread 模式
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from PySide6.QtCore import QObject, QThread, Signal, Slot

from models.analyzer import (
    ClusteringParams, ClusteringResult,
    run_phenotype_clustering, optimize_k_selection
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ClusteringWorker(QObject):
    """
    聚类计算工作对象

    使用 QObject + moveToThread 模式，通过 @Slot 装饰的方法启动计算。
    支持多品种独立聚类。
    """

    # 自定义信号
    progress_signal = Signal(str)  # 进度信息
    result_signal = Signal(object)  # ClusteringResult 对象（单品种）
    multi_result_signal = Signal(object)  # Dict[str, ClusteringResult]（多品种）
    error_signal = Signal(str)  # 错误信息
    finished_signal = Signal()  # 完成信号

    def __init__(self, df: pd.DataFrame, params: ClusteringParams,
                 variety_groups: Optional[List[str]] = None,
                 merge_all: bool = False):
        """
        初始化工作对象

        Args:
            df: 包含选中特征列和 original_db_index 的 DataFrame
            params: 聚类参数
            variety_groups: 品种列表，None 表示单品种模式
            merge_all: 是否合并所有品种聚类
        """
        super().__init__()
        self._df = df
        self._params = params
        self._variety_groups = variety_groups or ['未知品种']
        self._merge_all = merge_all
        self._is_cancelled = False

    @Slot()
    def run(self):
        """
        执行聚类计算

        此方法在工作线程中执行，通过信号与主线程通信。
        支持多品种独立聚类。
        """
        try:
            if self._merge_all:
                # 全品种合并聚类模式（需求9.3.2）
                self._run_merged_group()
            elif len(self._variety_groups) > 1:
                # 多品种独立聚类模式
                self._run_multi_group()
            else:
                # 单品种聚类模式（需求9.3.1默认）
                self._run_single_group()

        except Exception as e:
            self.error_signal.emit(f"聚类计算出错: {str(e)}")
            self.finished_signal.emit()

    def _run_single_group(self):
        """单品种聚类"""
        self.progress_signal.emit("正在准备数据...")

        # 保留 original_db_index 以便后续更新数据库
        original_indices = self._df['original_db_index'].values
        features_df = self._df[self._params.features].copy()

        # 强制转换为数值类型
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        # 数据清洗
        self.progress_signal.emit("正在清洗数据...")
        valid_mask = features_df.notna().all(axis=1) & (features_df > 0).all(axis=1)
        clean_df = features_df[valid_mask].copy()
        clean_indices = original_indices[valid_mask.values]

        if len(clean_df) < 2:
            self.error_signal.emit("有效数据量不足，无法进行聚类")
            self.finished_signal.emit()
            return

        if self._is_cancelled:
            self.finished_signal.emit()
            return

        # 标准化
        self.progress_signal.emit("正在标准化数据...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(clean_df)

        if self._is_cancelled:
            self.finished_signal.emit()
            return

        # 确定 K 值
        if self._params.n_clusters is None:
            self.progress_signal.emit("正在自动优化 K 值...")
            k = optimize_k_selection(data_scaled, self._params.max_k)
        else:
            k = self._params.n_clusters

        if self._is_cancelled:
            self.finished_signal.emit()
            return

        # 执行聚类
        self.progress_signal.emit(f"正在执行 K={k} 的聚类...")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)

        # 构建结果 DataFrame
        result_df = pd.DataFrame({
            'original_db_index': clean_indices,
            'cluster_id': cluster_labels
        })

        # 还原聚类中心
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=self._params.features)
        centroids_df.index.name = 'cluster_id'

        if self._is_cancelled:
            self.finished_signal.emit()
            return

        # 构建结果对象
        result = ClusteringResult(
            df_with_clusters=result_df,
            centroids_df=centroids_df,
            k=k,
            n_samples=len(clean_df)
        )

        self.progress_signal.emit("聚类完成")
        self.result_signal.emit(result)
        self.finished_signal.emit()

    def _run_multi_group(self):
        """多品种独立聚类"""
        group_results: Dict[str, ClusteringResult] = {}
        total_groups = len(self._variety_groups)

        for i, group_name in enumerate(self._variety_groups):
            if self._is_cancelled:
                self.finished_signal.emit()
                return

            self.progress_signal.emit(f"正在处理品种 {i + 1}/{total_groups}: {group_name}")

            # 筛选当前品种的数据
            if 'variety_name' in self._df.columns:
                group_df = self._df[self._df['variety_name'] == group_name].copy()
            else:
                group_df = self._df.copy()

            if group_df.empty:
                continue

            # 聚类
            result = self._cluster_single_group(group_df, group_name)
            if result is not None:
                group_results[group_name] = result

            if self._is_cancelled:
                self.finished_signal.emit()
                return

        if not group_results:
            self.error_signal.emit("所有品种的有效数据量均不足")
            self.finished_signal.emit()
            return

        self.progress_signal.emit(f"多品种聚类完成，共处理 {len(group_results)} 个品种")
        self.multi_result_signal.emit(group_results)
        self.finished_signal.emit()

    def _cluster_single_group(self, df: pd.DataFrame, group_name: str) -> Optional[ClusteringResult]:
        """
        对单个品种执行聚类

        Args:
            df: 品种数据
            group_name: 品种名称

        Returns:
            ClusteringResult 或 None
        """
        original_indices = df['original_db_index'].values
        features_df = df[self._params.features].copy()

        # 强制转换为数值类型
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        # 数据清洗
        valid_mask = features_df.notna().all(axis=1) & (features_df > 0).all(axis=1)
        clean_df = features_df[valid_mask].copy()
        clean_indices = original_indices[valid_mask.values]

        if len(clean_df) < 2:
            return None

        # 标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(clean_df)

        # 确定 K 值
        if self._params.n_clusters is None:
            k = optimize_k_selection(data_scaled, self._params.max_k)
        else:
            k = self._params.n_clusters

        # 执行聚类
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)

        # 构建结果 DataFrame
        result_df = pd.DataFrame({
            'original_db_index': clean_indices,
            'cluster_id': cluster_labels
        })

        # 还原聚类中心
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=self._params.features)
        centroids_df.index.name = 'cluster_id'

        return ClusteringResult(
            df_with_clusters=result_df,
            centroids_df=centroids_df,
            k=k,
            n_samples=len(clean_df)
        )

    def _run_merged_group(self):
        """
        全品种合并聚类模式（需求9.3.2）

        所有品种数据合并执行一次 KMeans，然后按品种独立重编号 cluster_id。
        """
        group_results: Dict[str, ClusteringResult] = {}
        variety_names = self._df['variety_name'].dropna().unique() if 'variety_name' in self._df.columns else ['全部品种']

        self.progress_signal.emit(f"正在合并 {len(variety_names)} 个品种数据进行聚类...")

        # 1. 对所有数据执行统一聚类
        original_indices = self._df['original_db_index'].values
        features_df = self._df[self._params.features].copy()

        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        valid_mask = features_df.notna().all(axis=1) & (features_df > 0).all(axis=1)
        clean_df = features_df[valid_mask].copy()
        clean_indices = original_indices[valid_mask.values]

        if len(clean_df) < 2:
            self.error_signal.emit("有效数据量不足，无法进行聚类")
            self.finished_signal.emit()
            return

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(clean_df)

        if self._params.n_clusters is None:
            k = optimize_k_selection(data_scaled, self._params.max_k)
        else:
            k = self._params.n_clusters

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)

        # 2. 按品种分组，独立重编号 cluster_id
        for variety_name in variety_names:
            if self._is_cancelled:
                self.finished_signal.emit()
                return

            self.progress_signal.emit(f"正在处理品种: {variety_name}")

            # 筛选该品种的有效数据
            variety_mask = self._df['variety_name'].values[valid_mask.values] == variety_name
            variety_clean_indices = clean_indices[variety_mask]
            variety_cluster_labels = cluster_labels[variety_mask]

            if len(variety_clean_indices) == 0:
                continue

            # 该品种内重编号 cluster_id（从0开始连续）
            unique_labels = sorted(set(variety_cluster_labels))
            remap = {old: new for new, old in enumerate(unique_labels)}
            remapped_labels = [remap[l] for l in variety_cluster_labels]

            result_df = pd.DataFrame({
                'original_db_index': variety_clean_indices,
                'cluster_id': remapped_labels
            })

            # 聚类中心（使用统一 KMeans 的中心）
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            centroids_df = pd.DataFrame(centroids, columns=self._params.features)
            centroids_df.index.name = 'cluster_id'

            group_results[variety_name] = ClusteringResult(
                df_with_clusters=result_df,
                centroids_df=centroids_df,
                k=k,
                n_samples=len(variety_clean_indices)
            )

        if not group_results:
            self.error_signal.emit("所有品种的有效数据量均不足")
            self.finished_signal.emit()
            return

        self.progress_signal.emit(f"合并聚类完成，共处理 {len(group_results)} 个品种")
        self.multi_result_signal.emit(group_results)
        self.finished_signal.emit()

    def cancel(self):
        """取消计算"""
        self._is_cancelled = True
