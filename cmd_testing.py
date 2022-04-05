import sklearn
import tensorflow as tf
import numpy as np
import cv2
# from tensorflow import keras
# import matplotlib.pyplot as plt
# from PIL import Image
# import time
import os
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten , MaxPool2D , Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_path = 'img/train_img/'
# data = tf.keras.preprocessing.image_dataset_from_directory(image_path)


targets = {"1" : "Book" , "2" : "Pen"}

print(targets["1"])
# print(data)
# print(type(data))
# img = cv2.imread(image_path)
# img = cv2.resize(img , (330,330))
# print(img.shape)
# for roots, dirs, files in os.walk(image_path):
# 	for f in files:
# 		if f.endswith('.jpg'):
# 			path = os.path.join(roots , f)
# 			image = cv2.imread(path)
# 			image = cv2.resize(image , (330,330))
# 			input_arr = tf.keras.preprocessing.image.img_to_array(image)
# 			input_arr = np.array([input_arr])  # Convert single image to a batch.

# print(input_arr.shape)
# print(image)
# print(input_arr)