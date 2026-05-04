"""
聚类分析算法模块
提供 K-Means++ 聚类和 K 值自动优化功能
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


def optimize_k_selection(data_scaled, max_k=10):
    """
    通过肘部法则自动识别并返回最佳 K 值
    """
    n_samples = data_scaled.shape[0]

    # 【新增改进】：确保尝试的聚类数不大于样本总数
    # 通常 max_k 不应超过样本数减 1
    actual_max_k = min(max_k, n_samples - 1)

    if actual_max_k < 2:
        print(f"警告：样本量仅为 {n_samples}，无法进行手肘法分析，默认设定 K=2")
        return 2

    wcss = []
    # 使用调整后的 actual_max_k
    for i in range(1, actual_max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    # 自动寻找拐点
    kl = KneeLocator(range(1, actual_max_k + 1), wcss, curve='convex', direction='decreasing')
    best_k = kl.elbow if kl.elbow else 2  # 样本少时默认建议 2 类

    print(f"样本总数: {n_samples}, 自动搜索范围: 1-{actual_max_k}, 建议 K = {best_k}")
    return best_k


def run_phenotype_clustering(df, n_clusters=None, max_k=10):
    """
    针对植物表型数据的 K-Means++ 聚类算法实现
    """
    # 1. 特征列定义
    features = ['width', 'height', 'perimeter', 'area', 'similarity']

    # 2. 数据清洗 (剔除空值及物理意义非法的负值/零值)
    df_clean = df.dropna(subset=features).copy()
    df_clean = df_clean[(df_clean[features] > 0).all(axis=1)]

    if len(df_clean) < (n_clusters if n_clusters else 2):
        return None, None

    # 3. 标准化 (消除面积等大数值对相似度等小数值的权重压制)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean[features])

    # 4. K值确定
    if n_clusters is None:
        k = optimize_k_selection(data_scaled, max_k)

    # 5. 执行聚类
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df_clean['cluster_id'] = kmeans.fit_predict(data_scaled)

    # 6. 还原聚类中心（回原始量级用于生物学分析）
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features)
    centroids_df.index.name = 'cluster_id'

    return df_clean, centroids_df