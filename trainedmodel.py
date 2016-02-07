from __future__ import print_function

from IPython import embed
import numpy as np
import os


from keras.models import Sequential

model = Sequential()




model.load_weights('test.h5')
DIR = 'tests/'
numTest = len(os.listdir(DIR))
print str(numTest)
img_rows, img_cols = 64, 64
trainX = np.ndarray((numTest,img_rows,img_cols))