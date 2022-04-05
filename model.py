import numpy as np
import cv2
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten , MaxPool2D , Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , History

path = "img/train_img"
targets = np.array([])
i = 1

train = ImageDataGenerator(rescale=1./255,
                rotation_range=20,
                horizontal_flip=True,
                shear_range = 0.2,
                fill_mode = 'nearest',
                validation_split=0.4)

test = ImageDataGenerator(rescale = 1.0/255,
                        validation_split=0.4) 

train_df = train.flow_from_directory(path , target_size =(330,330), batch_size = 3, 
class_mode = 'binary' , color_mode = 'grayscale')

test_df = test.flow_from_directory(path , 
target_size =(330,330), batch_size = 3, class_mode = 'binary' , color_mode = 'grayscale')

# for roots, dirs , files in os.walk(path):
# 	# print(files)
# 	for f in files:
# 		if f.endswith(".jpg"):
# 			targets = np.append(targets , roots[-1])
# 			image = cv2.imread(os.path.join(roots, f))
# 			# print(image.shape)
# 			# print(">>> Img Array {} passed <<< \n".format(i))
# 			# time.sleep(0.2)
# 			# i += 1
# 			# image = np.append(image, img)
# 			# print(image)
history = History()
model = Sequential([
	keras.layers.Conv2D(64 , (3,3) , activation = 'relu' , input_shape = (330,330,1)),
	keras.layers.MaxPool2D(2,2),
	keras.layers.Dropout(0.3),
	keras.layers.BatchNormalization(),

	keras.layers.Conv2D(32 , (3,3) , activation = 'relu'),
	keras.layers.MaxPool2D(2,2),
	keras.layers.Dropout(0.3),

	keras.layers.Conv2D(16 , (3,3) , activation = 'relu'),
	keras.layers.MaxPool2D(2,2),
	keras.layers.Dropout(0.3),

	# keras.layers.Conv2D(16 , (3,3) , activation = 'relu'),
	# keras.layers.MaxPool2D(2,2),
	# keras.layers.Dropout(0.3),

	keras.layers.Flatten(),		
	keras.layers.Dense(64, activation = 'relu'),	
	keras.layers.Dense(32, activation = 'relu'),	
	keras.layers.Dropout(0.3),
	keras.layers.Dense( 3 , activation = 'softmax')			
	])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.000003)
											
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
optimizer = keras.optimizers.Adam(learning_rate=0.001) , metrics = ['accuracy'])

model.fit(train_df , epochs = 15 , validation_data = test_df, callbacks=[history])

model.save("trained_model4.h5")

print("Saved model at the dirctory")


# print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()