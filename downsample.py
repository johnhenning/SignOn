import cv2
import sys, os
import numpy as np

import glob

def downsample():
	count = 0
	for img in glob.glob("test/sampled/*.jpg"):
	    src = cv2.imread(img)
	    for num in range(0,5):
	    	out = cv2.pyrDown(src)
	    	src = out
	    	if num == 3:
	    		gray_image = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
	    		cv2.imwrite("test/output" + str(count) + ".jpg",gray_image)
	    count += 1

    	

downsample()