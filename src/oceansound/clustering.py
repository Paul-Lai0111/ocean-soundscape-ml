"""Unsupervised grouping of windows into soundscape classes.

Standardize -> PCA -> K-Means, choosing k by silhouette score. The same
recipe as the Academia Sinica ceramic-clustering project, applied to sound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    labels: np.ndarray            # cluster id per window
    k: int
    silhouette_by_k: dict[int, float]
    pca_coords: np.ndarray        # (n, n_components)
    explained_variance: np.ndarray
    loadings: pd.DataFrame        # feature x PC
    feature_names: list[str]


def cluster_features(df: pd.DataFrame, feature_cols: list[str], k_range=range(2, 9),
                     k: int | None = None, var_keep: float = 0.95, seed: int = 0) -> ClusterResult:
    """Cluster the rows of ``df`` using ``feature_cols``.

    If ``k`` is None, pick the k in ``k_range`` with the best silhouette score.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=var_keep, svd_solver="full", random_state=seed)
    Z = pca.fit_transform(X)
    if Z.shape[1] < 2:  # keep at least 2 PCs so the scatter plot works
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(X)

    n = len(df)
    ks = [kk for kk in k_range if 2 <= kk < n]
    sil: dict[int, float] = {}
    fits: dict[int, np.ndarray] = {}
    for kk in ks:
        lab = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(Z)
        sil[kk] = float(silhouette_score(Z, lab))
        fits[kk] = lab

    if k is None:
        k = max(sil, key=sil.get)
    labels = fits[k] if k in fits else KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)

    loadings = pd.DataFrame(
        pca.components_.T, index=feature_cols,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    return ClusterResult(labels, k, sil, Z, pca.explained_variance_ratio_, loadings, feature_cols)


def evaluate_against_labels(true_labels, cluster_labels) -> tuple[float, pd.DataFrame]:
    """Adjusted Rand index and a cluster-vs-truth contingency table."""
    ari = adjusted_rand_score(true_labels, cluster_labels)
    table = pd.crosstab(pd.Series(true_labels, name="true"),
                        pd.Series(cluster_labels, name="cluster"))
    return float(ari), table


def describe_clusters(df: pd.DataFrame, feature_cols: list[str], label_col: str = "cluster") -> pd.DataFrame:
    """Mean of each feature per cluster, plus cluster size."""
    out = df.groupby(label_col)[feature_cols].mean()
    out.insert(0, "n_windows", df.groupby(label_col).size())
    return out
