from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, Iterable, Tuple, Sequence, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels, label_name=None):
    """
    Compute and (optionally) print three common clustering‐quality metrics.

    Parameters
    ----------
    X : array‐like, shape (n_samples, n_features)
        The feature matrix (e.g. your scaled training set).
    labels : array‐like, shape (n_samples,)
        Cluster assignment for each sample.
    label_name : str, optional
        An identifier to print (e.g. 'k=4' or 'Agglomerative').

    Returns
    -------
    dict
        {
            'silhouette': float,
            'davies_bouldin': float,
            'calinski_harabasz': float
        }
    """
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)

    if label_name is not None:
        print(f"--- {label_name} ---")
    print(f"Silhouette Score:       {sil:.4f}")
    print(f"Davies–Bouldin Score:   {db:.4f}")
    print(f"Calinski–Harabasz Score:{ch:.4f}")

    return {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(
    data,
    feature,
    bins=50,
    xlim=None,
    ylim=None,
    kde=True,
    color=None,
    title=None
):
    """
    Plots the distribution of a feature with optional x/y axis limits.
    
    Parameters:
    - data: DataFrame or array
    - feature: str (column name) or array
    - bins: int, number of bins for the histogram
    - xlim: tuple (xmin, xmax) to limit x-axis
    - ylim: tuple (ymin, ymax) to limit y-axis
    - kde: bool, whether to overlay KDE (density curve)
    - color: color of histogram bars
    - title: str, optional plot title
    """
    plt.figure(figsize=(8, 5))
    
    if isinstance(feature, str):
        values = data[feature]
        label = feature
    else:
        values = feature
        label = "Feature"
    
    sns.histplot(values, bins=bins, kde=kde, color=color, edgecolor='black')
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    plt.xlabel(label)
    plt.ylabel('Count')
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Distribution of {label}")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def kmeans_silhouette(df, column1="PC1", column2="PC2", k_values=[2, 3, 4, 5, 6, 7]):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np  
    import seaborn as sns
    from sklearn.decomposition import PCA



    for i, k in enumerate(k_values):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df)

        # Create a copy of the original DataFrame
        pc_df = df.copy()

        # Add PC1 and PC2 as new columns
        pc_df[["PC1", "PC2"]] = pd.DataFrame(
            principal_components,
            columns=["PC1", "PC2"],
            index=df.index  # Ensure alignment by index
        )
        print(pca.explained_variance_ratio_)
        # Run KMeans
        km = KMeans(n_clusters=k, random_state=42)
        y_predict = km.fit_predict(df)
        centroids = km.cluster_centers_

        # Silhouette scores
        silhouette_vals = silhouette_samples(df, y_predict)
        avg_score = np.mean(silhouette_vals)

        # Silhouette plot
        y_lower = 0
        for j, cluster in enumerate(np.unique(y_predict)):
            cluster_silhouette_vals = silhouette_vals[y_predict == cluster]
            cluster_silhouette_vals.sort()
            y_upper = y_lower + len(cluster_silhouette_vals)

            ax[0].barh(range(y_lower, y_upper),
                    cluster_silhouette_vals,
                    height=1)
            ax[0].text(-0.03, (y_lower + y_upper) / 2, str(cluster + 1))
            y_lower = y_upper

        ax[0].axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax[0].set_yticks([])
        ax[0].set_xlim([-0.1, 1])
        ax[0].set_xlabel('Silhouette coefficient values')
        ax[0].set_ylabel('Cluster labels')
        ax[0].set_title('Silhouette plot for the various clusters')

        # Cluster visualization (scatter plot)
        ax[1].scatter(pc_df[column1], pc_df[column2], c=y_predict, cmap='Spectral', alpha=0.6)
        ax[1].scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=250)
        ax[1].set_xlabel('Petal length (cm)')
        ax[1].set_ylabel('Petal width (cm)')
        ax[1].set_title('Visualization of clustered data')

        # Final layout
        plt.suptitle(f'Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'Silhouette_analysis_{k}.jpg')
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil, sqrt
from typing import Sequence, Optional, Union, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil, sqrt
from typing import Sequence, Optional, Union


from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Sequence, Union


import pandas as pd
import numpy as np

def find_cluster_feature_zscore(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    feature_cols: Optional[Sequence[str]] = None,
    z_score_cutoff: float = 2.0
) -> pd.DataFrame:
    """
    For each numeric feature, computes the z-score of each cluster’s mean
    relative to the overall (global) mean and std, then returns those
    cluster–feature pairs whose |z_score| >= z_score_cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing numeric features and a cluster column.
    cluster_col : str
        Name of the column with cluster labels.
    feature_cols : sequence of str, optional
        Which features to test.  If None, uses all numeric columns except cluster_col.
    z_score_cutoff : float
        Minimum absolute z-score to flag a cluster–feature pair.

    Returns
    -------
    pd.DataFrame
        A table with columns [cluster_col, "feature", "z_score"], listing
        every feature in every cluster whose mean deviates from the global
        mean by at least z_score_cutoff standard deviations.
    """
    # 1. Select features
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number") \
                         .columns.difference([cluster_col])
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns to analyze.")

    # 2. Compute global means and stds (population std, ddof=0)
    global_mean = df[feature_cols].mean()
    global_std = df[feature_cols].std(ddof=0)

    # 3. Compute cluster means
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()

    # 4. Compute z-scores for each cluster and feature
    #    (broadcasting: subtract Series from each row of DataFrame)
    z_scores = (cluster_means - global_mean) / global_std

    # 5. Melt to long form and filter by cutoff
    z_long = (
        z_scores
        .reset_index()
        .melt(id_vars=cluster_col, var_name="feature", value_name="z_score")
    )
    outliers = z_long.loc[z_long["z_score"].abs() >= z_score_cutoff].copy()
    outliers = outliers.sort_values(by=["z_score"], key=lambda col: col.abs(), ascending=False)
    outliers.reset_index(drop=True, inplace=True)

    return outliers



def create_violinplots_by_cluster(
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
        cluster_col: str = "cluster",
        figsize: Union[tuple, None] = None,
        palette: str = "Set2",
        inner: str = "quart",
        outlier_multiplier: float = 3.0,
        savepath: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
    """
    Draws side-by-side violin plots of each numeric feature’s distribution,
    grouped by the categories in *cluster_col*, **excluding only very extreme outliers**.

    Outliers are defined as values beyond `outlier_multiplier × IQR` for that feature
    (within each cluster), and are removed before computing the density.
    The violin is also clipped to the data range (`cut=0`) and each
    violin is scaled to the same maximum width (`scale='width'`).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features and the cluster label column.
    feature_cols : sequence of str, optional
        Columns to plot. Defaults to **all numeric** columns except *cluster_col*.
    cluster_col : str, default 'cluster'
        Column that defines the cluster membership.
    figsize : tuple, optional
        Overall figure size (width, height) in inches.  If None, a square-ish
        layout is chosen automatically.
    palette : str, default 'Set2'
        Any seaborn-compatible palette name.
    inner : {'box', 'quart', 'point', 'stick', None}, default 'quart'
        How to represent the datapoints inside each violin.
    outlier_multiplier : float, default 3.0
        Multiplier for the IQR when defining outliers (e.g., 3×IQR removes only very extreme values).
    savepath : str or None
        If given, the figure is also written to this path (formats inferred
        from the extension).
    show : bool, default True
        Whether to immediately display the plot (use False inside scripts).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for further tweaking if desired.
    """
    # 1. Choose which feature columns to plot
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number") \
                         .columns.difference([cluster_col])
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns to plot.")

    # 2. Decide subplot grid size
    n = len(feature_cols)
    if figsize is None:
        side = ceil(n**0.5)
        figsize = (4 * side, 3 * side)
    rows = ceil(n / ceil(n**0.5))
    cols = ceil(n / rows)

    # 3. Build the grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # 4. Draw each violin, filtering only very extreme outliers per cluster
    for ax, feature in zip(axes, feature_cols):
        df_plot = df[[cluster_col, feature]].dropna()

        # exclude extreme outliers per cluster
        def _filter_extreme(sub):
            Q1 = sub[feature].quantile(0.25)
            Q3 = sub[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - outlier_multiplier * IQR
            upper = Q3 + outlier_multiplier * IQR
            return sub[sub[feature].between(lower, upper)]

        df_plot = df_plot.groupby(cluster_col, group_keys=False).apply(_filter_extreme)

        sns.violinplot(
            data=df_plot,
            x=cluster_col,
            y=feature,
            palette=palette,
            inner=inner,
            cut=0,            # clip violin to data min/max
            scale='width',    # make all violins the same max width
            ax=ax
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel(feature)

    # Remove any extra (unused) axes
    for ax in axes[n:]:
        ax.remove()

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig



import pandas as pd
from typing import Sequence, Optional

def compute_cluster_zscores(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    cluster_col: str = "cluster"
) -> pd.DataFrame:
    """
    Compute, for each cluster, the z-score of each feature’s mean against the global distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Your data, containing numeric features and a cluster-label column.
    feature_cols : sequence of str, optional
        Which feature columns to include. Defaults to all numeric columns except `cluster_col`.
    cluster_col : str, default 'cluster'
        Column name holding the cluster labels.

    Returns
    -------
    zscores : pd.DataFrame
        Indexed by cluster, with one column per feature.  Each entry is
        (mean_in_cluster − global_mean) / global_std.
    """
    # 1. Decide which features to include
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number") \
                         .columns.drop(cluster_col)

    # 2. Global mean & std
    global_means = df[feature_cols].mean()
    global_stds  = df[feature_cols].std(ddof=0)  # population std by default; change ddof if you want sample std

    # 3. Per‑cluster means
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()

    # 4. Z‑score computation
    zscores = (cluster_means - global_means) / global_stds

    return zscores


# def compute_cluster_zscores(
#     df: pd.DataFrame,
#     feature_cols: Optional[Sequence[str]] = None,
#     cluster_col: str = "cluster"
# ) -> pd.DataFrame:
#     """
#     Compute, for each cluster, the z-score of each feature’s mean against the global distribution,
#     and plot bar charts for each cluster.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Your data, containing numeric features and a cluster-label column.
#     feature_cols : sequence of str, optional
#         Which feature columns to include. Defaults to all numeric columns except `cluster_col`.
#     cluster_col : str, default 'cluster'
#         Column name holding the cluster labels.

#     Returns
#     -------
#     zscores : pd.DataFrame
#         Indexed by cluster, with one column per feature. Each entry is
#         (mean_in_cluster − global_mean) / global_std.
#     """
#     # 1. Decide which features to include
#     if feature_cols is None:
#         feature_cols = df.select_dtypes(include="number") \
#                          .columns.drop(cluster_col)

#     # 2. Global mean & std
#     global_means = df[feature_cols].mean()
#     global_stds  = df[feature_cols].std(ddof=0)  # population std

#     # 3. Per-cluster means
#     cluster_means = df.groupby(cluster_col)[feature_cols].mean()

#     # 4. Z-score computation
#     zscores = (cluster_means - global_means) / global_stds

#     # 5. Plotting
#     for cluster in zscores.index:
#         plt.figure(figsize=(10, 5))
#         zscores.loc[cluster].plot(kind='bar')
#         plt.title(f"Z-scores for Cluster {cluster}")
#         plt.xlabel("Feature")
#         plt.ylabel("Z-score")
#         plt.axhline(0, color='gray', linestyle='--')
#         plt.tight_layout()
#         plt.show()

#     return zscores

def dbscan_grid_with_stats(
    X_embed: np.ndarray | pd.DataFrame,
    df_original: pd.DataFrame,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
    metric: str = "euclidean",
    n_jobs: int | None = -1,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[float, int], pd.DataFrame]]:
    """
    Sweep over (eps, min_samples); for every setting compute
      • number of clusters,
      • proportion of noise points,
      • silhouette score (noise removed, requires ≥2 clusters),
      • counts of z-score magnitudes in [1,2), [2,3), ≥3,
      • per-cluster z-scores of the original features.

    Returns
    -------
    summary_df : DataFrame
        One row per parameter pair with all diagnostics.
    zscore_map : dict
        {(eps, min_samples) : DataFrame_of_zscores}
    """
    results: list[Dict[str, Any]] = []
    zscore_map: Dict[Tuple[float, int], pd.DataFrame] = {}

    for eps in eps_values:
        for min_samples in min_samples_values:

            db = DBSCAN(eps=eps, min_samples=min_samples,
                        metric=metric, n_jobs=n_jobs)
            labels = db.fit_predict(X_embed)

            # scalar diagnostics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = np.count_nonzero(labels == -1)
            outlier_pct = 100 * n_outliers / len(labels)

            sil = np.nan
            if n_clusters >= 2:
                mask = labels != -1
                if mask.any():
                    sil = silhouette_score(X_embed[mask], labels[mask])

            # per-cluster z-scores
            df_tmp = df_original.copy()
            df_tmp["cluster"] = labels
            z_df = compute_cluster_zscores(
                df_tmp, feature_cols=feature_cols, cluster_col="cluster"
            )
            zscore_map[(eps, min_samples)] = z_df

            # magnitude counts
            abs_z = z_df.abs()
            count_1_2 = int(((abs_z >= 1) & (abs_z < 2)).sum().sum())
            count_2_3 = int(((abs_z >= 2) & (abs_z < 3)).sum().sum())
            count_gt3 = int((abs_z >= 3).sum().sum())

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'outlier_pct': outlier_pct,
                'silhouette': sil,
                'count_z_1_2': count_1_2,
                'count_z_2_3': count_2_3,
                'count_z_gt3': count_gt3,
            })

    summary_df = pd.DataFrame(results).sort_values(
        ['silhouette', 'outlier_pct'], ascending=[False, True]
    ).reset_index(drop=True)

    return summary_df.round(3), zscore_map





from pyclustertend import hopkins
import numpy as np

def hopkins_test(X, n_trials: int = 10, verbose: bool = True) -> float:
    """
    Compute the Hopkins statistic multiple times to assess clustering tendency.

    Parameters
    ----------
    X         : array-like, shape (n_samples, n_features)
                The data matrix to test (e.g., UMAP or PCA embedding).
    n_trials  : int, default=10
                Number of repeated Hopkins evaluations.
    verbose   : bool, default=True
                If True, print all individual scores and their mean.

    Returns
    -------
    mean_score : float
                 The mean Hopkins score across trials.
    """
    scores = [hopkins(X, X.shape[0]) for _ in range(n_trials)]
    mean_score = float(np.mean(scores))

    if verbose:
        print("All Hopkins scores:", np.round(scores, 3))
        print("Mean Hopkins score:", np.round(mean_score, 3))

    return mean_score



import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def cluster_zscore_dataframe(
        df: pd.DataFrame,
        cluster_col: str = "Cluster",
        features: list[str] | None = None,
        ddof: int = 0,
        rename_map: dict[str, str] | None = None
    ) -> pd.DataFrame:
    """
    Return a DataFrame whose rows are clusters and whose columns are features,
    containing the z-score of each cluster’s mean relative to the overall mean
    of the whole data set.  Optionally rename the feature columns.

    Z-score = (cluster_mean − overall_mean) / overall_std

    Parameters
    ----------
    df : pd.DataFrame
        The table that already contains a column with cluster labels.
    cluster_col : str, default="Cluster"
        Name of the column holding cluster labels.
    features : list[str] | None, default=None
        Which feature columns to include.  If None, all numeric columns except
        `cluster_col` are used.
    ddof : int, default=0
        Degrees-of-freedom argument forwarded to `pandas.DataFrame.std`.
    rename_map : dict[str, str] | None, default=None
        If provided, keys are original feature names, values are new column names.

    Returns
    -------
    pd.DataFrame
        Index → cluster labels  
        Columns → (possibly renamed) features  
        Cells  → z-scores
    """
    # 1️⃣ Pick the features to analyse
    if features is None:
        features = (
            df.select_dtypes(include="number")
              .columns
              .drop(cluster_col)
              .tolist()
        )

    # 2️⃣ Global statistics
    overall_mean = df[features].mean()
    overall_std  = df[features].std(ddof=ddof).replace(0, np.nan)

    # 3️⃣ Per-cluster means
    cluster_means = (
        df.groupby(cluster_col, observed=True)[features]
          .mean()
          .astype(float)
    )

    # 4️⃣ Z-scores
    zscores = (cluster_means - overall_mean) / overall_std

    # 5️⃣ Optional renaming
    if rename_map:
        zscores = zscores.rename(columns=rename_map)

    return zscores



import numpy as np
import matplotlib.pyplot as plt

def plot_clusters_by_feature_matplotlib(
        zscores_df: pd.DataFrame,
        feature_labels: dict[str, str] | None = None,
        cluster_labels: dict[int, str] | None = None
    ):
    """
    Plot mean z-score bars grouped by cluster for each feature, with optional
    renaming of features and clusters.

    Parameters
    ----------
    zscores_df : pd.DataFrame
        Index   → cluster labels (ints)
        Columns → feature names (str)
        Values  → mean z-scores
    feature_labels : dict[str, str] | None
        Mapping from original column names → display names.
    cluster_labels : dict[int, str] | None
        Mapping from original cluster ints → display names (e.g. archetypes).
    """
    # 1️⃣ Prepare feature display names
    features = zscores_df.columns.tolist()
    display_features = (
        [feature_labels.get(f, f) for f in features]
        if feature_labels else
        features
    )

    # 2️⃣ Prepare cluster display names
    clusters = zscores_df.index.tolist()
    display_clusters = (
        [cluster_labels.get(c, str(c)) for c in clusters]
        if cluster_labels else
        [str(c) for c in clusters]
    )

    # 3️⃣ Set up bar positions
    x = np.arange(len(features))
    n_clusters = len(clusters)
    width = 0.8 / n_clusters

    # 4️⃣ Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (c, label) in enumerate(zip(clusters, display_clusters)):
        values = zscores_df.loc[c].values
        ax.bar(x + i*width, values, width, label=label)

    # 5️⃣ Decoration
    ax.set_xticks(x + width*(n_clusters-1)/2)
    ax.set_xticklabels(display_features, rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Z-score")
    ax.set_title("Mean Z-Score by Cluster for Each Feature")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
