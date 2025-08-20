import matplotlib.pyplot as plt
import numpy as np
import umap
from itertools import product
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE   
import seaborn as sns
import pandas as pd 
from sklearn.manifold import TSNE


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_PCA(df, cluster_labels=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)

    plt.figure(figsize=(8, 6))

    if cluster_labels is not None:
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=cluster_labels, cmap="tab10", alpha=0.6, edgecolor='k'
        )
        plt.legend(*scatter.legend_elements(), title="Cluster")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolor='k')

    plt.title("PCA Projection")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.show()


def plot_tsne_perplexities(df, perplexities=[5, 30, 50, 100], cluster_labels=None, random_state=42):
    plt.figure(figsize=(4 * len(perplexities), 4))

    for i, perp in enumerate(perplexities):
        X_tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state).fit_transform(df)

        plt.subplot(1, len(perplexities), i + 1)

        if cluster_labels is not None:
            scatter = plt.scatter(
                X_tsne[:, 0], X_tsne[:, 1],
                c=cluster_labels, cmap="tab10", s=10
            )
            if i == 0:
                plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10)

        plt.title(f"t-SNE (perplexity={perp})")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()



def plot_umap_grid(X, labels=None, param_grid=None, figsize=(15, 12), random_state=42):
    """
    Visualizes UMAP embeddings in a grid with different combinations of parameters.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels: array-like, shape (n_samples,), optional cluster labels for coloring
    - param_grid: dict with keys as parameter names and values as lists to test
        e.g., {"n_neighbors": [5, 15], "min_dist": [0.1, 0.5]}
    - figsize: tuple, overall figure size
    - random_state: int, for reproducibility
    """

    # Defaults
    if param_grid is None:
        param_grid = {
            "n_neighbors": [5, 15, 30],
            "min_dist": [0.1, 0.5, 0.9],
        }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    combinations = list(product(*param_values))
    n_rows = len(param_grid[param_names[0]])
    n_cols = len(param_grid[param_names[1]])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle("UMAP Grid Visualization", fontsize=16, y=1.02)

    for idx, (combo, ax) in enumerate(zip(combinations, axes.flat)):
        params = dict(zip(param_names, combo))

        reducer = umap.UMAP(
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            n_components=params.get("n_components", 2),
            random_state=random_state,
        )

        embedding = reducer.fit_transform(X)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels if labels is not None else "gray",
            cmap="Spectral",
            s=10,
            alpha=0.7,
        )

        title = ", ".join(f"{k}={v}" for k, v in params.items())
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
