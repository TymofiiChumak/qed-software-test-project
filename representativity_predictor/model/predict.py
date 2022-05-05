import mlflow
import click

import numpy as np

from .data_storage import download_data
from .common import log_errors

mlflow.tensorflow.autolog(log_models=False)


def _predict(model_reference, evaluattion_features):
    model = mlflow.keras.load_model(model_reference)
    if model.input.shape[1] != evaluattion_features.shape[1]:
        raise ValueError(
            "model was trained on objects with different dimensions: "
            f"{model.input.shape[1]} != {evaluattion_features.shape[1]}"
        )
    return model.predict(evaluattion_features)


def store_prediction_result(result: np.ndarray):
    result_filename = "prediction_result.csv"
    with open(result_filename, mode='w+') as dump_file:
        np.savetxt(dump_file, result, delimiter="\n")
    mlflow.log_artifact(result_filename)


mlflow.set_experiment("predict")


@click.command()
@click.argument('model_reference', type=click.STRING)
@click.argument('predict_data_key', type=click.STRING)
def predict(model_reference, predict_data_key):

    with mlflow.start_run(), log_errors():
        prediction_features = download_data(predict_data_key)
        result = _predict(model_reference, prediction_features)
        store_prediction_result(result)


if __name__ == "__main__":
    predict()
