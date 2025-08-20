from typing import Tuple, Optional, Dict, Union

import numpy as np
from sklearn.decomposition import PCA
import umap
from matplotlib import pyplot as plt
def scale_dataframe_standard(df):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # --- pick up every numeric dtype, not just float64/int64 ---
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError(
            "scale_dataframe_standard found no numeric columns to scale. "
            f"Detected dtypes: {df.dtypes.unique()}"
        )

    # --- scale them ---
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[num_cols])

    # --- put them back into a copy of the df ---
    df_scaled = df.copy()
    df_scaled[num_cols] = scaled_vals

    return df_scaled, scaler


def remove_outliers_iqr(df, columns=None, factor=5):
    import pandas as pd
    """
    Removes rows from a DataFrame where any specified numeric column has outliers,
    as defined by the IQR method.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to filter on. If None, defaults to all numeric columns.
    - factor: multiplier for the IQR (1.5 is standard; 3.0 is more conservative)
    
    Returns:
    - filtered_df: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    # Determine which columns to filter on
    if columns is None:
        # Use all numeric columns if no specific columns are provided
        columns = df_clean.select_dtypes(include=["float64", "int64"]).columns.tolist()
    else:
        # Retain only the columns that exist in df and are numeric
        columns = [col for col in columns if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col])]
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Filter the DataFrame for the current column
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def preprocess_data(df):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import random

    # Separate feature types
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        # ("scaler", StandardScaler())  # Optional
    ])

    # Categorical pipeline (no encoder)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Fit the pipeline and transform the data
    df_preprocessed = preprocessor.fit_transform(df)

    # Combine column names: numeric and categorical as-is
    all_columns = list(num_cols) + list(cat_cols)

    # Return as DataFrame with original index preserved
    return pd.DataFrame(df_preprocessed, columns=all_columns, index=df.index)



def pca_umap_reduction(
    X: np.ndarray,
    *,
    pca_components: Optional[int] = None,
    pca_variance: float = 0.95,
    umap_components: int = 2,
    random_state: Union[int, None] = 42,
    umap_kwargs: Optional[Dict] = None,
    use_umap: bool = True,
    
) -> np.ndarray:
    """
    Reduce X by PCA, then (optionally) UMAP.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    pca_components : int or None
        If None, select by explained variance (pca_variance).
    pca_variance : float
        Fraction of variance to keep if pca_components is None.
    umap_components : int
        Dimensionality of UMAP embedding.
    random_state : int or None
    umap_kwargs : dict or None
        Passed to umap.UMAP(...)
    use_umap : bool
        If False, skip UMAP and return the PCA projection.

    Returns
    -------
    embedding : ndarray of shape
        - (n_samples, umap_components) if use_umap
        - (n_samples, pca_components or inferred) otherwise
    """
    # PCA step
    pca = PCA(
        n_components=pca_components or pca_variance,
        svd_solver="full",
        random_state=random_state,
    ).fit(X)
    X_pca = pca.transform(X)

    from matplotlib import pyplot as plt
    import random
    if not use_umap:
        return X_pca
    
    # Visualization of PCA with the first two components
    if pca.n_components_ >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, cmap='Spectral', alpha=0.7)
        plt.title("PCA Projection (First Two Components)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(label="Density")
        plt.show()

    # UMAP step
    umap_kwargs = umap_kwargs or {}
    um = umap.UMAP(
        n_components=umap_components,
        random_state=random_state,
        **umap_kwargs,
    ).fit(X_pca)
    import matplotlib.pyplot as plt

    # Choose two random components for visualization
    if umap_components > 2:
        comp1, comp2 = random.sample(range(umap_components), 2)
        plt.figure(figsize=(8, 6))
        plt.scatter(um.embedding_[:, comp1], um.embedding_[:, comp2], s=5, cmap='Spectral', alpha=0.7)
        plt.title(f"UMAP Projection (Random Components {comp1 + 1} and {comp2 + 1})")
        plt.xlabel(f"Component {comp1 + 1}")
        plt.ylabel(f"Component {comp2 + 1}")
        plt.colorbar(label="Density")
        plt.show()


    return um.embedding_

