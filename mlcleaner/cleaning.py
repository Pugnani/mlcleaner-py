import pandas as pd
import numpy as np

#Clean large DataFrame with vectorized operations
def clean(df: pd.DataFrame, 
          impute_strategy: str = 'median', 
          outlier_method: str = 'iqr') -> pd.DataFrame:
    """
    Efficiently clean a large DataFrame: vectorized operations for speed, fill missing values, remove outliers.
    """
    df = df.drop_duplicates(ignore_index=True)
    df = impute_missing(df, strategy=impute_strategy)

    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df)
    else:
        raise ValueError("outlier_method must be 'iqr' or 'zscore'")

    return df

# Fill missing values in numeric columns (vectorized)
def impute_missing(
    df: pd.DataFrame,
    strategy: str = 'median',
    columns: list = None
) -> pd.DataFrame:
    """
    Efficiently fill missing values in numeric columns.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    if strategy == 'median':
        values = df[columns].median()
        df[columns] = df[columns].fillna(values)
    elif strategy == 'mean':
        values = df[columns].mean()
        df[columns] = df[columns].fillna(values)
    elif strategy == 'mode':
        values = df[columns].mode().iloc[0]
        df[columns] = df[columns].fillna(values)
    elif strategy == 'ffill':
        df[columns] = df[columns].fillna(method='ffill')
    elif strategy == 'bfill':
        df[columns] = df[columns].fillna(method='bfill')
    else:
        raise ValueError(f"Invalid strategy '{strategy}' for imputation")
    return df

# Remove outliers using IQR method (vectorized)
def remove_outliers_iqr(df: pd.DataFrame, columns: list = None, multiplier: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask = (df[columns] >= lower) & (df[columns] <= upper)
    mask = mask.all(axis=1)
    return df[mask]

# Remove outliers using Z-score method (vectorized)
def remove_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
    columns: list = None
) -> pd.DataFrame:
    """
    Efficiently remove rows where numeric columns have Z-score beyond a threshold.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std(ddof=0))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]
