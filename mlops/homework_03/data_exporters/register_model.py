from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

@data_exporter
def export_data(
    data, **kwargs
) -> Tuple[DictVectorizer, LinearRegression]:

    dv, lr = data

    with mlflow.start_run():
        with open("dict_vectorizer.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dict_vectorizer.bin", artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr, artifact_path="model")
    return dv, lr

