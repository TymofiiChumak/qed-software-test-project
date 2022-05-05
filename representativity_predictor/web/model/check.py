from dataclasses import dataclass
import json
import os
from typing import Optional
import mlflow
from datetime import datetime
from flask import jsonify
from mlflow.exceptions import MlflowException

from .app import model_app as app
from .common import get_train_run


def _print_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000).isoformat()


@dataclass
class ErrorInfo:
    error: str
    traceback: str


def _fetch_error_info(run_id: str) -> Optional[ErrorInfo]:
    try:
        errors_filename = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/errors.json"
        )
        with open(errors_filename, "r") as source:
            error_data = ErrorInfo(**json.load(source))
        os.remove(errors_filename)
        return error_data
    except MlflowException:
        return None


@app.route("/", methods=["GET"])
@app.route("/<model_id>", methods=["GET"])
def check_model_status(**kwargs):
    run = get_train_run(kwargs.get("model_id"))

    response = {
        "id": run.info.run_id,
        "start_time": _print_timestamp(run.info.start_time),
        "status": run.info.status,
    }

    end_time = getattr(run.info, "end_time", None)

    if end_time:
        response["end_time"] = _print_timestamp(end_time)

    if run.info.status == "FAILED":
        error_info = _fetch_error_info(run.info.run_id)
        if error_info:
            response["error"] = error_info.error

    try:
        response["progress"] = (
            (float(run.data.metrics["submodels"]) + 1)
            / float(run.data.params["batch_number"])
        )
    except KeyError:
        response["progress"] = 0

    return jsonify(response)
