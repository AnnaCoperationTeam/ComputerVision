import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import time

cam = cv2.VideoCapture(1)
# cam2 = cv2.VideoCapture(0)
cam.set(3,480)
cam.set(4,520)
cam.set(10,100)

# cam2.set(3,480)
# cam2.set(4,520)
# cam2.set(10,100)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = keras.models.load_model("trained_model3.h5")
# model.build()
# model.summary()
path = 'img/test_img'
test_images = []
targets = []
ker = np.ones((5,5),np.uint8)
test = keras.preprocessing.image.ImageDataGenerator(rescale = (1./255))
test_df = test.flow_from_directory('img/test_img' , target_size =(330,330), batch_size = 3, class_mode = 'binary')


# for roots, dirs,files in os.walk(path):
# 	for f in files:
# 		if f.endswith('.jpg'):
# 			path = os.path.join(roots , f)
# 			targets.append(roots[-1])
# 			img = cv2.resize(cv2.imread(path) , (330,330))
# 			img = cv2.imread(path) / 255
# 			test_images.append(img)
# 			# print(path)
# test_images = np.array(test_images)
# targets = np.array(targets)
# # images = images.reshape(85,220,220,3)
# model.evaluate(test_images , targets)
# result = model.predict(test_df)
# print(result)

targets = {"0" : "Rock" , "1" : "Paper" , "2" : "Scissors"}
# model.evaluate(test_images , targets)
while True:

	ret , img = cam.read()
	# ret2 , img2 = cam2.read()
	# img = cv2.flip(img, 1)

	gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY_INV,31,2)

	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	# faces=face_cascade.detectMultiScale(gray,scaleFactor=1.10,minNeighbors=5 , flags = cv2.CASCADE_SCALE_IMAGE)
	cv2.imshow("Gray" , img)
	cv2.imshow("th" , opening)


	# ret, th = cv2.threshold(gray, 100 , cv2.THRESH_BINARY)

	# for x,y,w,h in faces:
		# image=cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0),2)
	# cv2.imshow("Cam2" , img2)

	gray_img = cv2.resize(opening , (330,330))
	gray_img = gray_img.astype('float32')
	gray_img = gray_img / 255
	gray_img = tf.keras.preprocessing.image.img_to_array(gray_img)
	gray_img = np.array([gray_img])
	# gray_img = gray_img[50: , 100:]
	# print(gray_img.shape)
	# time.sleep(1.5)
	result = model.predict(gray_img)
	# print(result[0][0])
	# result = np.argmax(result)
	cv2.putText(img , "Rock :{} Paper :{} Scissors :{}".format(str(result[0][0]),
	str(result[0][1]),
	str(result[0][2])) , 
	(0 , 25), cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,(0,255,255) , 1)

	cv2.imshow("Cam",img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
