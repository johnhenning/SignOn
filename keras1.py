'''Train a simple convnet on the MNIST dataset.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from IPython import embed
import numpy as np


import glob
np.random.seed(1337)  # for reproducibility

from PIL import Image 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 15
nb_classes = 3
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

numTrain = 3118

trainX = np.ndarray((numTrain,img_rows,img_cols))
trainY = np.genfromtxt ('data.csv', delimiter=",", dtype = np.int8)

count = 0
for img in glob.glob("training/*.jpg"):
	data = np.asarray(Image.open(img))
	trainX[count] = data
	count += 1

random = np.random.randint(numTrain, size=300).tolist()

testX = trainX[random]
testY = trainY[random]

np.delete(trainX,random)
np.delete(trainY,random)

print(trainX.shape)
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = (trainX,trainY), (testX,testY)
#embed()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

model.save_weights('test.h5', overwrite=True)
