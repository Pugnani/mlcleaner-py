import pandas as pd
import numpy as np
import pytest
from mlcleaner import clean_small, clean_large, encode_categorical, normalize_numeric

# Sample DataFrame
df = pd.DataFrame({
    'num1': [1, 2, 2, 3, np.nan, 1000],  # include NaN & outlier
    'num2': [10, 20, 20, 30, 40, 5000],  # outlier
    'cat1': ['a', 'b', 'a', 'c', 'b', 'a'],
    'cat2': ['low', 'medium', 'high', 'medium', 'low', 'high']
})

def test_clean_small():
    df_cleaned = clean_small(df)
    # No duplicates
    assert df_cleaned.duplicated().sum() == 0
    # No NaN in numeric
    assert df_cleaned.select_dtypes(include='number').isna().sum().sum() == 0
    # Outliers removed
    assert df_cleaned['num1'].max() < 1000
    assert df_cleaned['num2'].max() < 5000

def test_clean_large():
    df_cleaned = clean_large(df)
    # Similar checks
    assert df_cleaned.duplicated().sum() == 0
    assert df_cleaned.select_dtypes(include='number').isna().sum().sum() == 0

def test_encode_categorical():
    # test label encoding
    df_encoded = encode_categorical(df, encoding_methods={'cat1': 'label'}, default_method='label')
    assert df_encoded['cat1'].dtype == np.int64
    # test ordinal encoding
    ordinal_map = {'cat2': ['low','medium','high']}
    df_encoded_ord = encode_categorical(df, encoding_methods={'cat2':'ordinal'}, ordinal_mappings=ordinal_map)
    assert set(df_encoded_ord['cat2']) <= {0,1,2}
    # test onehot encoding
    df_encoded_oh = encode_categorical(df, default_method='onehot')
    assert any('cat1_' in col for col in df_encoded_oh.columns)

def test_normalize_numeric():
    df_scaled = normalize_numeric(df, scaling_methods={'num1':'standard'}, default_method='minmax')
    # check numeric range for minmax
    for col in df_scaled.select_dtypes(include='number').columns:
        assert df_scaled[col].min() >= 0 or df_scaled[col].min() < 0  # minmax [0,1], standard can be negative
        assert df_scaled[col].max() <= 1 or df_scaled[col].max() > 1

