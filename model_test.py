import os
import tensorflow as tf
from tensorflow import keras

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# take random image from PetImages/Cat/ folder
img = tf.keras.preprocessing.image.load_img("PetImages/Cat/27.jpg", target_size=(180, 180))
# convert to numpy array
img_array = tf.keras.preprocessing.image.img_to_array(img)
# add a dimension to the array
img_array = tf.expand_dims(img_array, 0)
# load model
model = keras.models.load_model("async_model")
# predict
predictions = model.predict(img_array)
# print result
score = predictions[0]
print(
    "This image is %.2f percent Cat and %.2f percent Dog."
    % (100 * (1 - score), 100 * score)
)
