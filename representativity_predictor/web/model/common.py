
from typing import Optional
from flask import jsonify, Request
import mlflow
from mlflow.exceptions import MlflowException
from werkzeug.exceptions import BadRequest, NotFound
import os
from uuid import uuid4
import boto3


DATA_STORAGE_BUCKET = "data"


s3 = boto3.resource(
    "s3",
    endpoint_url=f'http://{os.environ["MINIO_HOST"]}:9000',
)


def ensure_bucket(bucket_name: str):
    bucket = s3.Bucket(bucket_name)
    if bucket.creation_date is None:
        return s3.create_bucket(Bucket=bucket_name)
    else:
        return bucket


def upload_data(file) -> str:
    bucket = ensure_bucket(DATA_STORAGE_BUCKET)
    filename = str(uuid4())
    bucket.upload_fileobj(file, filename)
    return filename


def get_train_run(run_id: Optional[str], check_status: bool = False):
    try:
        if run_id:
            run = mlflow.get_run(run_id)
            if check_status and run.info.status != 'FINISHED':
                raise BadRequest(response=jsonify({
                    "error": f"Model has wrong state'{run.info.status}'"
                }))
            return run
        else:
            if check_status:
                filter_string = "attribute.status = 'FINISHED'"
            else:
                filter_string = ""
            return mlflow.search_runs(
                order_by=["attribute.start_time DESC"],
                filter_string=filter_string,
                max_results=1,
                output_format='list',
            )[0]
    except (MlflowException, IndexError):
        raise NotFound(response=jsonify({
            "error": "Available model not found"
        }))


def save_file_from_request(request: Request) -> str:
    if 'file' not in request.files:
        raise BadRequest(response=jsonify({
            "error": "Request must contains file",
        }))

    return upload_data(request.files['file'])
