from typing import Dict, List, Union, Optional, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
import pandas as pd
import scipy

CATEGORICAL_FEATURES = ['PU_DO']
NUMERICAL_FEATURES = ['trip_distance']

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def combine_features(df: Union[List[Dict], DataFrame]) -> Union[List[Dict], DataFrame]:
    if isinstance(df, DataFrame):
        df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    elif isinstance(df, list) and len(df) >= 1 and isinstance(df[0], dict):
        arr = []
        for row in df:
            row['PU_DO'] = str(row['PULocationID']) + '_' + str(row['DOLocationID'])
            arr.append(row)
        return arr
    return df

def vectorize_features(
    training_set: pd.DataFrame
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, DictVectorizer]:
    dv = DictVectorizer()

    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    return X_train, dv

def select_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    if features:
        columns += features

    return df[columns]

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = kwargs.get('target', 'duration')
    split_on_feature = kwargs.get('split_on_feature', 'lpep_pickup_datetime')

    df = combine_features(df)
    df = select_features(df, features=[split_on_feature, target])

    X, dv = vectorize_features(df)
    y: Series = df[target]

    # fit model
    reg = LinearRegression().fit(X, y)
    predictions = reg.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"intercept: {reg.intercept_}")
    print(f"rmse: {rmse}")

    return dv, reg