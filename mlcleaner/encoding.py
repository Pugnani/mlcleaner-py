import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode categorical columns with different methods
def encode_categorical(
    df: pd.DataFrame,
    encoding_methods: dict = None,
    ordinal_mappings: dict = None,
    default_method: str = None
) -> pd.DataFrame:
    """
    Encode categorical columns with different methods per column and optional default method. Specify by the User.
    :param df: DataFrame to process
    :param encoding_methods: dict {col_name: method} where method is 'label', 'onehot', 'ordinal'
    :param ordinal_mappings: dict for ordinal encoding {col_name: [category order]}
    :param default_method: method to encode remaining categorical columns not in encoding_methods
    :return: DataFrame with encoded columns
    """
    df = df.copy()
    if encoding_methods is None:
        encoding_methods = {}
    cat_cols = df.select_dtypes(include='object').columns
    processed_cols = set()
    # Apply specific methods
    for col, method in encoding_methods.items():
        if col not in df.columns:
            continue
        if method == 'label':
            df[col] = pd.factorize(df[col])[0]  # faster than LabelEncoder
        elif method == 'onehot':
            # collect columns for onehot to do all at once later
            pass  # handled later
        elif method == 'ordinal':
            df[col] = df[col].map({cat: i for i, cat in enumerate(ordinal_mappings[col])})
        processed_cols.add(col)
    # Default method for remaining
    remaining_cols = [c for c in cat_cols if c not in processed_cols]
    if default_method == 'label':
        for col in remaining_cols:
            df[col] = pd.factorize(df[col])[0]
    elif default_method == 'onehot':
        df = pd.get_dummies(df, columns=remaining_cols, drop_first=True)
    return df
