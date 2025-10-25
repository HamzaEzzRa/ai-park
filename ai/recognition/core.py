from typing import Optional, Tuple

import keras
from keras import layers

recognition_model: Optional[keras.Model] = None


def get_recognition_model(
    input_shape: Optional[Tuple[int, int, int]] = (16, 16, 4),
    num_group_types: int = 2,
    max_group_size: int = 3,
) -> Optional[keras.Model]:
    global recognition_model
    if recognition_model is not None:
        return recognition_model

    inputs = keras.Input(shape=input_shape, name="visitor_sprite")

    x = layers.Rescaling(1.0 / 255.0, name="scale_0_1")(inputs)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    group_type_output = layers.Dense(
        num_group_types,
        activation="softmax",
        name="group_type"
    )(x)
    group_size_output = layers.Dense(
        max_group_size,
        activation="softmax",
        name="group_size"
    )(x)

    recognition_model = keras.Model(
        inputs=inputs,
        outputs={
            "group_type": group_type_output,
            "group_size": group_size_output,
        },
        name="visitor_recognition_model"
    )
    return recognition_model


def train_recognition_model(
    model: keras.Model,
    *,
    train_data,
    val_data,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Compile and train the recognition model on visitor data.

    If ``train_data`` is omitted the helper builds a :class:`VisitorDataset`
    using image dimensions inferred from the model input.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    losses = {
        "group_type": keras.losses.SparseCategoricalCrossentropy(),
        "group_size": keras.losses.SparseCategoricalCrossentropy(),
    }
    metrics = {
        "group_type": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        "group_size": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    }
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
    )
    return model, history
