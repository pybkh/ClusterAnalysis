"""
聚类分析算法模块
提供 K-Means++ 聚类和 K 值自动优化功能
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


@dataclass
class ClusteringParams:
    """聚类参数"""
    features: List[str]  # 特征列列表
    n_clusters: Optional[int] = None  # K 值，None 表示自动优化
    max_k: int = 10  # 自动优化时的最大 K 值


@dataclass
class ClusteringResult:
    """聚类结果"""
    df_with_clusters: pd.DataFrame  # 包含 cluster_id 的 DataFrame
    centroids_df: pd.DataFrame  # 聚类中心 DataFrame
    k: int  # 实际使用的 K 值
    n_samples: int  # 有效样本数


def optimize_k_selection(data_scaled, max_k=10):
    """
    通过肘部法则自动识别并返回最佳 K 值
    """
    n_samples = data_scaled.shape[0]

    # 确保尝试的聚类数不大于样本总数
    actual_max_k = min(max_k, n_samples - 1)

    if actual_max_k < 2:
        print(f"警告：样本量仅为 {n_samples}，无法进行手肘法分析，默认设定 K=2")
        return 2

    wcss = []
    for i in range(1, actual_max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    # 自动寻找拐点
    kl = KneeLocator(range(1, actual_max_k + 1), wcss, curve='convex', direction='decreasing')
    best_k = kl.elbow if kl.elbow else 2

    print(f"样本总数: {n_samples}, 自动搜索范围: 1-{actual_max_k}, 建议 K = {best_k}")
    return best_k


def run_phenotype_clustering(df: pd.DataFrame, features: Optional[List[str]] = None,
                              n_clusters: Optional[int] = None, max_k: int = 10) -> Optional[ClusteringResult]:
    """
    针对植物表型数据的 K-Means++ 聚类算法实现

    Args:
        df: 包含特征列的 DataFrame
        features: 要使用的特征列列表，默认使用全部特征列
        n_clusters: 指定的 K 值，None 表示自动优化
        max_k: 自动优化时的最大 K 值

    Returns:
        ClusteringResult 或 None（数据不足时）
    """
    # 1. 特征列定义
    if features is None:
        features = ['width', 'height', 'perimeter', 'area', 'similarity']

    # 验证特征列是否存在
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        return None

    # 2. 数据清洗 (剔除空值及物理意义非法的负值/零值)
    df_clean = df.dropna(subset=available_features).copy()
    df_clean = df_clean[(df_clean[available_features] > 0).all(axis=1)]

    if len(df_clean) < (n_clusters if n_clusters else 2):
        return None

    # 3. 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean[available_features])

    # 4. K值确定
    if n_clusters is None:
        k = optimize_k_selection(data_scaled, max_k)
    else:
        k = n_clusters

    # 5. 执行聚类
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df_clean['cluster_id'] = kmeans.fit_predict(data_scaled)

    # 6. 还原聚类中心（回原始量级用于生物学分析）
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=available_features)
    centroids_df.index.name = 'cluster_id'

    return ClusteringResult(
        df_with_clusters=df_clean,
        centroids_df=centroids_df,
        k=k,
        n_samples=len(df_clean)
    )
