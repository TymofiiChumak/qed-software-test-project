from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance_matrix
import tensorflow as tf

import mlflow
import click

from .data_storage import download_data
from .common import log_errors


@dataclass
class Config:
    nearest_neighbours: int
    batch_number: int
    learning_rate: float
    train_epochs: int


def calculate_representativity(
        features: np.ndarray, config: Config) -> np.ndarray:
    distances = distance_matrix(features, features)
    k = config.nearest_neighbours
    nearest = np.argpartition(distances, k+1)[::, :k+1]
    k_nearest_distances = np.take_along_axis(distances, nearest, axis=-1)
    return 1 / (1 + (np.sum(k_nearest_distances, axis=1) / k))


def prepare_simple_model(
        inputs, features, config: Config) -> tf.keras.models.Model:
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(features))

    normalizer_layer = normalizer(inputs)
    linear_layer = tf.keras.layers.Dense(units=1)(normalizer_layer)

    model = tf.keras.models.Model(inputs=inputs, outputs=linear_layer)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mean_absolute_error'
    )
    return model


def train_model(model, features, target, config: Config):
    return model.fit(
        features,
        target,
        epochs=config.train_epochs,
        verbose=0,
        validation_split=0.2,
    )


def create_ensebled_model(data, config: Config):
    data_parted = np.array_split(data, config.batch_number)
    inputs = tf.keras.layers.Input(shape=(data.shape[1],))

    models = []
    for part_idx, features_part in enumerate(data_parted):
        target = calculate_representativity(features_part, config)
        model = prepare_simple_model(inputs, features_part, config)
        train_model(model, features_part, target, config)
        models.append(model)
        mlflow.log_metric("submodels", part_idx)

    for model_idx, model in enumerate(models):
        for layer in model.layers[1:]:
            layer.trainable = False
            layer._name = f"ensemble_{model_idx}_{layer.name}"

    ensemble_outputs = [model.output for model in models]
    average = tf.keras.layers.average(ensemble_outputs)
    model = tf.keras.Model(inputs=inputs, outputs=average)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


mlflow.tensorflow.autolog(log_models=False)
mlflow.set_experiment("train")


@click.command()
@click.argument('train_data_key', type=click.STRING)
@click.option(
    '--nearest-neighbours',
    type=click.INT,
    default=5,
)
@click.option(
    '--batch-number',
    type=click.INT,
    default=10,
)
@click.option(
    '--learning-rate',
    type=click.FLOAT,
    default=0.01,
)
@click.option(
    '--train-epochs',
    type=click.INT,
    default=100,
)
def prepare_model(train_data_key, **kwargs):
    config = Config(**kwargs)

    with mlflow.start_run(), log_errors():
        train_features = download_data(train_data_key)
        model = create_ensebled_model(train_features, config)
        mlflow.keras.log_model(model, "ensebled_model")


if __name__ == "__main__":
    prepare_model()
