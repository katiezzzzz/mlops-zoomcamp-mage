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

def vectorize_features(
    df: pd.DataFrame
) -> Tuple[scipy.sparse.csr_matrix, DictVectorizer]:
    dv = DictVectorizer()

    train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    return X_train, dv

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[DictVectorizer, LinearRegression]:
    target = kwargs.get('target', 'duration')

    X, dv = vectorize_features(df)
    y: Series = df[target]

    # fit model
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"intercept: {lr.intercept_}")
    print(f"rmse: {rmse}")

    return dv, lr