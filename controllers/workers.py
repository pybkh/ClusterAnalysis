"""
工作线程模块
负责在后台执行耗时的聚类计算任务
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from PySide6.QtCore import QThread, Signal

from models.analyzer import optimize_k_selection
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ClusteringWorker(QThread):
    """
    聚类计算工作线程
    继承 QThread，在后台执行 K-Means 聚类算法
    """

    # 自定义信号
    progress_signal = Signal(str)  # 进度信息
    result_signal = Signal(object, object)  # (df_with_cluster_id, centroids_df)
    error_signal = Signal(str)  # 错误信息
    finished_signal = Signal()  # 完成信号

    def __init__(self, df: pd.DataFrame, selected_features: List[str],
                 n_clusters: Optional[int] = None, max_k: int = 10):
        """
        初始化工作线程

        Args:
            df: 包含选中特征列和 original_db_index 的 DataFrame
            selected_features: 用户选择的特征列名列表
            n_clusters: 指定的 K 值，None 表示自动优化
            max_k: 自动优化时的最大 K 值
        """
        super().__init__()
        self._df = df
        self._selected_features = selected_features
        self._n_clusters = n_clusters
        self._max_k = max_k
        self._is_cancelled = False

    def cancel(self):
        """取消计算"""
        self._is_cancelled = True

    def run(self):
        """
        执行聚类计算

        注意：由于原 run_phenotype_clustering 函数内部硬编码了特征列，
        这里我们直接使用 sklearn 实现聚类逻辑，以支持用户自定义特征选择。
        """
        try:
            self.progress_signal.emit("正在准备数据...")

            # 保留 original_db_index 以便后续更新数据库
            original_indices = self._df['original_db_index'].values
            features_df = self._df[self._selected_features].copy()

            # 数据清洗：剔除空值和负值
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
            if self._n_clusters is None:
                self.progress_signal.emit("正在自动优化 K 值...")
                k = optimize_k_selection(data_scaled, self._max_k)
            else:
                k = self._n_clusters

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
            centroids_df = pd.DataFrame(centroids, columns=self._selected_features)
            centroids_df.index.name = 'cluster_id'

            if self._is_cancelled:
                self.finished_signal.emit()
                return

            self.progress_signal.emit("聚类完成")
            self.result_signal.emit(result_df, centroids_df)
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"聚类计算出错: {str(e)}")
            self.finished_signal.emit()
