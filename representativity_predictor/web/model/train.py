import os
import mlflow
from flask import jsonify, request
from werkzeug.exceptions import BadRequest
from .app import model_app as app
from .common import save_file_from_request


@app.route("/", methods=["POST"])
def train_model():
    mlflow.set_experiment("train")
    if mlflow.active_run() is not None:
        return BadRequest(response=jsonify({
            "error": "There is other run in progress"
        }))

    filename = save_file_from_request(request)

    parameters = {"train_data_key": filename}
    parameters_keys = ["nearest_neighbours", "batch_number"]
    for parameter_key in parameters_keys:
        if parameter_key in request.form:
            parameters[parameter_key] = request.form[parameter_key]

    scheduled_run = mlflow.run(
        os.environ["JOB_PATH"],
        entry_point="train",
        parameters=parameters,
        docker_args={"network": os.environ["WORKER_CONTAINERS_NETWORK"]},
        synchronous=False,
    )
    return jsonify({"model_id": scheduled_run.run_id}), 200
