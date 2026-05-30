import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


model = load_model("models\cnn.keras")

img= r"data\clothing-dataset-small\validation\shoes\5e577f40-dd22-4b40-9827-ce6cae5ac3fd.jpg"

img = tf.keras.utils.load_img(img, target_size=(224,224))
img = tf.keras.utils.img_to_array(img)
img = tf.expand_dims(img,0)
prediction = model.predict(img)

result = tf.nn.softmax(prediction)

print(result)
class_names = ['dress','hat','longsleeve','outwear','pants','shirt','shoes','shorts','skirt','t-shirt']

print(class_names[np.argmax(result)])

