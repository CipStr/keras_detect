import os
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers
# from matplotlib import pyplot as plt
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Function that makes a keras model
# We start the model with the data_augmentation preprocessor, followed by a Rescaling layer.
# The Rescaling layer rescales the input image to have values between 0 and 1.
# We include a Dropout layer before the final classification layer.
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"],
        'ps': ["localhost:12346", "localhost:12348"],
        'chief': ["localhost:12347"]
    },
    'task': {'type': 'chief', 'index': 0}
})

# augment dataset
data_augmentation = keras.Sequential(
     [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(16, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def create_dataset():
    # generate dataset
    image_size = (180, 180)
    batch_size = 2
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    # print number of images in each dataset
    print("Number of training images before augmentation: " + str(len(train_ds)))
    # print type of train_ds
    print("Type of train_ds: " + str(type(train_ds)))

    # convert train_ds from BatchDataset to Dataset
    train_ds = train_ds.unbatch()
    # print type of train_ds
    print("Type of train_ds: " + str(type(train_ds)))


    # is train_ds an iterator?
    print("Is train_ds an iterator? " + str(isinstance(train_ds, tf.compat.v1.data.Iterator)))
    # is train_ds an iterable?
    print("Is train_ds an iterable? " + str(isinstance(train_ds, tf.compat.v1.data.Iterator)))

    train_ds = train_ds.prefetch(buffer_size=2)
    val_ds = val_ds.prefetch(buffer_size=2)
    return train_ds, val_ds, image_size


# resolve cluster
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if cluster_resolver.task_type in ("worker", "ps"):
    # wait for chief to finish
    # Start a TensorFlow server and wait.
    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    # print server status and port
    print("Server address: " + server.target)
    print("Server status: " + str(server.server_def))
    print("Server port: " + str(server.target.split(":")[-1]))
    server.join()
else:
    strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
    train_ds, val_ds, image_size = create_dataset()

    # create dataset
    # train_ds, val_ds, data_augmentation, image_size = create_dataset()
    # open a strategy scope
    with strategy.scope():
        model = make_model(input_shape=image_size + (3,), num_classes=2)
        # keras.utils.plot_model(model, show_shapes=True)
        # get model summary
        model.summary()
        # run training
        # callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["acc"],
        )
        # get compile status
        print("Compile status: " + str(model.optimizer))

    epochs = 2
    steps_per_epoch = 4
    history = model.fit(
        train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=val_ds,
    )

    # print history
    print(history.history)
