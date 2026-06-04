# import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
# import cv2 as cv

from colorthief import  ColorThief
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image
import os 


# model = load_model("models\cnn.keras")

img_path= r"test.jpg"

# img = tf.keras.utils.load_img(img, target_size=(224,224))
# img = tf.keras.utils.img_to_array(img)
# img = tf.expand_dims(img,0)
# prediction = model.predict(img)

# result = tf.nn.softmax(prediction)

# print(result)
# class_names = ['dress','hat','longsleeve','outwear','pants','shirt','shoes','shorts','skirt','t-shirt']

# print(class_names[np.argmax(result)])

# img = cv.imread(img_path)
# img = cv.resize(img, (700,400), interpolation=cv.INTER_LINEAR)

# h, w, _ = np.shape(img)
# data = np.reshape(img,(h * w,3))
# data = np.float32(data)

# clu = 3
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10,1.0)
# flags = cv.KMEANS_RANDOM_CENTERS



# cv.imshow('Image' ,img)

# cv.waitKey(0)


img = 'janu3.jpg'
# ct = ColorThief(img)

# dom = ct.get_color(quality=1)

# plt.imshow([[dom]])
# plt.axis('off')
# plt.show()
# plt.imshow(plt.imread(img))
# plt.axis('off')
# plt.show()



# pal = ct.get_palette(color_count= 5)

# plt.imshow([[pal[i] for i in range(5)]])
# plt.axis('off')
# plt.show()

# # for color in pal :
# #     print(color)

# print(np.array(pal))

img_path = r'data\clothing-dataset-small\validation\shorts\3e203f7a-5b90-4b27-9303-a5634a80f015.jpg'
img = Image.open(img_path)

no_bg = remove(img)
plt.imshow(no_bg)
plt.show()

plt.imshow(plt.imread('trans.png'))
plt.show()

no_bg_loc = 'trans.png'

no_bg.save(no_bg_loc)
# no_bg_img = Image.('trans.png')

ct = ColorThief(img_path)

dom = ct.get_color(quality=1)

plt.imshow([[dom]])
plt.axis('off')
plt.show()


# pal = ct.get_palette(color_count=3)

# plt.imshow([[pal[i] for i in range(3)]])
# plt.axis('off')
# plt.show()