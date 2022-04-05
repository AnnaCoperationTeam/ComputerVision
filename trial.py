import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

path_r = "rock.MOV"
path_p = "paper.MOV"
path_s = "scissors.MOV"

_r= "rock"
_p = "paper"
_s = "scissors"

vid = cv2.VideoCapture(path_s)
vid.set(3 , 480)
vid.set(4 , 480)
i = 0

path = "img/train_img/3"
i = 0
os.chdir(path)

ker = np.ones((5,5),np.uint8)
while(vid.isOpened()):
    ret , img = vid.read()
    img= cv2.resize(img , (480,480))
    # plt.imshow(img)
    # plt.show()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY_INV,31,2)

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    gray = opening.astype("float32")
    gray -= gray.mean()
    gray /= gray.std()

    # canny = cv2.Canny(edge , 100 ,200)
    
    cv2.imshow("video" , img)
    cv2.imshow("Gray" , gray)
    cv2.imshow("Closing" , closing)
    cv2.imshow("Open" , opening)    

    # cv2.imshow("th" , th)

    # cv2.imwrite(_s + str(i) + ".jpg" , gray)
    # print(gray.shape)
    # i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
