import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import time

model = keras.models.load_model("trained_model4.h5")

path = "S_test2.jpg"

img = cv2.imread(path)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                     cv2.THRESH_BINARY_INV,31,2)

# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

gray_img = cv2.resize(blurred , (330,330))
gray_img = gray_img.astype('float32')
gray_img = gray_img / 255
gray_img = tf.keras.preprocessing.image.img_to_array(gray_img)
gray_img = np.array([gray_img])

# print(gray_img.shape)
res = model.predict(gray_img)
print(res)