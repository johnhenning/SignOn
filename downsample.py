import cv2
import sys, os
import numpy as np

import glob

count = 1317
for img in glob.glob("training/undownsampled/*.jpg"):
    src = cv2.imread(img)
    for num in range(0,5):
    	out = cv2.pyrDown(src)
    	src = out
    	if num == 3:
    		gray_image = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    		cv2.imwrite("training/output" + str(count) + ".jpg",gray_image)
    count += 1

    	






