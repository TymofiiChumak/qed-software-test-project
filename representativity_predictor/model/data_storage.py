import os
from tempfile import NamedTemporaryFile
import numpy as np
import boto3


DATA_STORAGE_BUCKET = "data"


class DataLoadError(Exception):
    pass


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


ensure_bucket(os.environ["ARTIFACTS_BUCKET"])


def parse_data(source_file: str) -> np.ndarray:
    try:
        return np.genfromtxt(source_file, delimiter=",")
    except ValueError as error:
        raise DataLoadError("can not parse source file") from error


def download_data(train_data_file: str) -> np.ndarray:
    bucket = ensure_bucket(DATA_STORAGE_BUCKET)
    with NamedTemporaryFile(delete=False) as temp_file:
        bucket.download_file(
            train_data_file,
            temp_file.name,
        )
        return parse_data(temp_file.name)
