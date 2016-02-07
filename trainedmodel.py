

from IPython import embed
import numpy as np
import os


from PIL import Image 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import glob


DIR = 'test/'
numTest = len(os.listdir(DIR)) - 2


nb_classes = 3

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 7


img_rows, img_cols = 64, 64





#embed()
#print predictions

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.load_weights('test.hdf5')

model.compile(loss='categorical_crossentropy', optimizer='adadelta')




import cv2
import sys, os




def downsample(src):
	count = 0
	out = cv2.resize(src,dsize=(1820,1024))
	out = out[:,398:-398]
	for num in range(0,4):
		src = cv2.pyrDown(out)
		out = src	
	gray_image = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
	print (gray_image.shape)
	return gray_image	





def show_webcam(mirror=False):
  	cam = cv2.VideoCapture(0)
  	testX = np.ndarray((1,img_rows,img_cols))
  	#fgbg = cv2.BackgroundSubtractorMOG()
	
  	A = np.array(Image.open("A.jpg"))
  	P = np.array(Image.open("P.jpg"))
  	I = np.array(Image.open("I.jpg"))

	while True:
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, 1)

		#fgmask = fgbg.apply(img)
		gray = downsample(img)
		testX[0] = gray
		atestX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
		predictions = model.predict(testX)
		print predictions
		print "predictions.shape = ", predictions.shape
		cv2.imshow('testing', gray)
		#cv2.imshow('backgroundsub', fgmask)
		cv2.imshow('my webcam', img)
		if (predictions == 1).any():
			prediction = np.where(predictions == 1)[1][0]
		print "prediction = ", prediction
		if prediction == 0:
			cv2.imshow('Letter', A)
		elif prediction == 1:
			cv2.imshow('Letter', P)
		elif prediction == 2:
			cv2.imshow('Letter', I)

		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	cv2.destroyAllWindows()

def main():
	show_webcam(mirror=True)

if __name__ == '__main__':
	main()


