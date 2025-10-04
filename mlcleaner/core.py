import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler


def clean_small(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean the Dataframe removing duplicates, handling NaN values and outliers.
    """
    df = df.drop_duplicates().copy()  # copy to avoid SettingWithCopyWarning
    # Put the median in the NaN values
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Remove outliers using IQR method
    mask = pd.Series(True, index=df.index)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= df[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    df = df.loc[mask].copy()
    return df
#For big datasets
def clean_large(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()  # Don't copy immediatly
    num_cols = df.select_dtypes(include='number').columns
    # Fill NaN nin a vector way
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Outlier removal using vectorized mask
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ((df[num_cols] >= (Q1 - 1.5*IQR)) & (df[num_cols] <= (Q3 + 1.5*IQR))).all(axis=1)
    return df.loc[mask]

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
