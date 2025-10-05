import pandas as pd
import numpy as np
import pytest

from mlcleaner.cleaning import clean, impute_missing, remove_outliers_iqr, remove_outliers_zscore
from mlcleaner.encoding import encode_categorical
from mlcleaner.scaling import normalize_numeric

# Sample DataFrame
df = pd.DataFrame({
    'num1': [1, 2, 2, 3, np.nan, 1000],
    'num2': [10, 20, 20, 30, 40, 5000],
    'cat1': ['a', 'b', 'a', 'c', 'b', 'a'],
    'cat2': ['low', 'medium', 'high', 'medium', 'low', 'high']
})

def test_clean():
    df_cleaned = clean(df)
    assert df_cleaned.duplicated().sum() == 0
    assert df_cleaned.select_dtypes(include='number').isna().sum().sum() == 0
    assert df_cleaned['num1'].max() < 1000
    assert df_cleaned['num2'].max() < 5000

def test_impute_missing():
    df_with_nan = df.copy()
    df_with_nan.loc[0, 'num1'] = np.nan
    df_imputed = impute_missing(df_with_nan, strategy='median')
    # Check no NaN remains
    assert df_imputed['num1'].isna().sum() == 0
    # Check median imputation works
    median_val = df_with_nan['num1'].median()
    assert df_imputed.loc[0, 'num1'] == median_val

def test_remove_outliers_zscore():
    df_no_outliers = remove_outliers_zscore(df, threshold=3.0)
    # Check extreme values are removed
    assert df_no_outliers['num1'].max() < 1000
    assert df_no_outliers['num2'].max() < 5000

def test_remove_outliers_iqr():
    df_no_outliers = remove_outliers_iqr(df)
    assert df_no_outliers['num1'].max() < 1000
    assert df_no_outliers['num2'].max() < 5000

def test_encode_categorical():
    df_encoded = encode_categorical(df, encoding_methods={'cat1': 'label'}, default_method='label')
    assert df_encoded['cat1'].dtype == np.int64

def test_normalize_numeric():
    df_scaled = normalize_numeric(df, scaling_methods={'num1':'standard'}, default_method='minmax')
    for col in df_scaled.select_dtypes(include='number').columns:
        assert not df_scaled[col].isna().any()
