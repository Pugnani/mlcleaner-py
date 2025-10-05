import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Normalize numeric columns with different scaling methods
def normalize_numeric(
    df: pd.DataFrame,
    scaling_methods: dict = None,
    default_method: str = None
) -> pd.DataFrame:
    """
    Normalize numeric columns with different scaling methods per column and optional default. The User specifies the column and the type of scaling.
    
    :param df: DataFrame to process
    :param scaling_methods: dict {col_name: method} where method is 'standard', 'minmax', 'robust'
    :param default_method: method to scale remaining numeric columns
    :return: DataFrame with scaled numeric columns
    """
    df = df.copy()
    if scaling_methods is None:
        scaling_methods = {}
    num_cols = df.select_dtypes(include='number').columns
    processed_cols = set()
    # group columns per method
    method_to_cols = {}
    for col, method in scaling_methods.items():
        method_to_cols.setdefault(method, []).append(col)
        processed_cols.add(col)
    # apply scaling per method group
    for method, cols in method_to_cols.items():
        if method == 'standard':
            df[cols] = StandardScaler().fit_transform(df[cols])
        elif method == 'minmax':
            df[cols] = MinMaxScaler().fit_transform(df[cols])
        elif method == 'robust':
            df[cols] = RobustScaler().fit_transform(df[cols])
    # default method for remaining numeric cols
    remaining_cols = [c for c in num_cols if c not in processed_cols]
    if default_method:
        if default_method == 'standard':
            df[remaining_cols] = StandardScaler().fit_transform(df[remaining_cols])
        elif default_method == 'minmax':
            df[remaining_cols] = MinMaxScaler().fit_transform(df[remaining_cols])
        elif default_method == 'robust':
            df[remaining_cols] = RobustScaler().fit_transform(df[remaining_cols])
    return df
