# Representativity predictor

Simple service to predict 'representattivity' of objects.

## Initial Configuraiton

Service required docker with docker compose installed on host.

Before first run of service some additional 
actions must be performed:

### 1. Prepare image for jobs

Image must be present on host where service is running

```bash
$ docker build . -t mlflow-executor:latest
```

### 2. Prepare environment file

Fill `.env` file with required values

```bash
$ cp env_example .eenv
$ vi .env
```

`MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` should be
filled with proper values. Example:

```
MINIO_ROOT_USER=AKIAIOSFODNN7EXAMPLE
MINIO_ROOT_PASSWORD=wJalrXUtnFEMI/K7MDENG/bPxRfiCYE
```

## Runing

To start service use following command:

```bash
$ docker compose up -d
```

## Usage

API use csv files as data source format. Files
hadn't contain headers. Symbol `","` should be
used s delimiter.

To creeate random test data following command can be used:

```bash
$ python -c 'import numpy as np;np.savetxt(<FILE_NAME>, np.random.rand(<OBJECT_COUNT>, <OBJECT_DIMENSION>), delimiter=",")'
```

### Start model trainig process

```bash
$ curl -L -XPOST 'http://0.0.0.0:5000/model' -F file=@<TRAIN_DATA>.csv
{"model_id":"574a97e860044bd5a9b94f177c6ade7b"}
```

### Check model training process status

```bash
$ curl -L 'http://0.0.0.0:5000/model'
# or $ curl -L 'http://0.0.0.0:5000/model/3acc540e0d354d8fb08fed8b40c7c69f'
{"id":"a8ea04a96eae4ea28fcddefd850702b4","progress":0.1,"start_time":"2022-05-05T17:48:50.315000","status":"RUNNING"}
```

### Predict representativity of object 

```bash
$ curl -L -XPUT 'http://0.0.0.0:5000/model/predict' -F file=@<EVALUATE_DATA>.csv -o result.csv
# or $ curl -L -XPUT 'http://0.0.0.0:5000/model/3acc540e0d354d8fb08fed8b40c7c69f/predict' -F file=@<EVALUATE_DATA>.csv -o result.csv
```