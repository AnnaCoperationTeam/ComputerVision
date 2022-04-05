import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
cam = cv2.VideoCapture(1)
cam.set(3,480)
cam.set(4,520)
cam.set(10 , 100)

_r= "rock"
_p = "paper"
_s = "scissors"


path = "img/test_img/3"
i = 0
os.chdir(path)

while True:
	state , img = cam.read()

	# img = increase_brightness(img , value=100)
	# img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
	# img = cv2.flip(img, 1)
	blur = cv2.GaussianBlur(img , (3,3), cv2.BORDER_DEFAULT)
	gray_img = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

	cv2.putText(img , "Current amount of images : {}".format(i) , (250,25) ,
	 cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,0,0) , 1, cv2.LINE_AA)

	ret , th = cv2.threshold(gray_img , 140 , 255 ,  cv2.THRESH_BINARY)
	ker = np.ones((5,5),np.uint8)
	morp = cv2.morphologyEx(th , cv2.MORPH_OPEN , kernel = ker)
	edge = cv2.morphologyEx(morp , cv2.MORPH_GRADIENT , kernel = ker)

	canny = cv2.Canny(edge, 100,200)


	cv2.imshow("capturing" , img)
	cv2.imshow("thresh" , th)

	cv2.imshow("gray" , gray_img)
	cv2.imshow("Gray_capture" , edge)

	# cv2.imshow("Blur" , blur)
	train_img = np.array([img])

	gray_img = cv2.resize(canny , (480,480))

	# cv2.imwrite(_s + str(i) + ".jpg" , canny)
	# plt.imshow(train_img)
	# plt.show()
	# i += 1
	# if i == 125:
	# 	break

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		# gray_img = cv2.resize(canny , (480,480))
		# cv2.imwrite(_r + str(i) + ".jpg" , canny)
