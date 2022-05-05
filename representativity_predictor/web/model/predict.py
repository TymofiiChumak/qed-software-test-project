

import os
import mlflow
from flask import jsonify, request, send_file
from mlflow.exceptions import MlflowException
from werkzeug.exceptions import InternalServerError
from .app import model_app as app
from .common import get_train_run, save_file_from_request


@app.route("/predict", methods=["PUT"])
@app.route("/<model_id>/predict", methods=["PuT"])
def predict(**kwargs):
    mlflow.set_experiment("train")
    train_run = get_train_run(kwargs.get("model_id"), check_status=True)

    filename = save_file_from_request(request)

    mlflow.set_experiment("predict")

    parameters = {
        "model_reference": f"runs:/{train_run.info.run_id}/ensebled_model",
        "predict_data_key": filename,
    }

    run = mlflow.run(
        os.environ["JOB_PATH"],
        entry_point="predict",
        parameters=parameters,
        docker_args={"network": os.environ["WORKER_CONTAINERS_NETWORK"]},
    )

    try:
        results_filename = mlflow.artifacts.download_artifacts(
            f"runs:/{run.run_id}/prediction_result.csv"
        )
        return send_file(results_filename)
    except MlflowException:
        return InternalServerError(response=jsonify({
            "error": "Can not get predict result"
        }))
