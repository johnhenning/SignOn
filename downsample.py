import cv2
import sys, os
import numpy as np

import glob

import cv2


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

def show_webcam(mirror=False):
  cam = cv2.VideoCapture(0)
	while True:
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, 1)
		cv2.imshow('my webcam', img)
		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	cv2.destroyAllWindows()

def main():
	show_webcam(mirror=True)

if __name__ == '__main__':
	main()

