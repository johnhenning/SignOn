from __future__ import print_function

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
#embed()
print (numTest)

nb_classes = 3

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 7




img_rows, img_cols = 64, 64

testX = np.ndarray((numTest,img_rows,img_cols))
embed()

count = 0
for img in glob.glob("tests/*.jpg"):
	data = np.array(Image.open(img))
	testX[count] = data
	count += 1

#embed()
testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)

embed()
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


predictions = model.predict(testX)
print (predictions)
