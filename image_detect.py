import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import json


# Function that makes a keras model
# We start the model with the data_augmentation preprocessor, followed by a Rescaling layer.
# The Rescaling layer rescales the input image to have values between 0 and 1.
# We include a Dropout layer before the final classification layer.
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
def make_model(input_shape, num_classes, data_augmentation):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
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

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # generate dataset
    image_size = (180, 180)
    batch_size = 32

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
    print("Number of validation images before augmentation: " + str(len(val_ds)))

    # data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    # configure dataset for performance
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    #open a strategy scope
    with strategy.scope():
        model = make_model(input_shape=image_size + (3,), num_classes=2, data_augmentation=data_augmentation)
        keras.utils.plot_model(model, show_shapes=True)
        # get model summary
        model.summary()
        # run training
        epochs = 2
        # callbacks
        callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        ]
        model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["acc"],
        )
    history = model.fit(
        train_ds, validation_split=0.1, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    # print training finised
    print("Training finished")
    # plot training history-accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # save plot to file
    plt.savefig('accuracy.png')
    plt.clf()

    # plot training history-loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # save plot to file
    plt.savefig('loss.png')
    # save model in saved_model format
    model.save("saved_model")


if __name__ == "__main__":
    main()
