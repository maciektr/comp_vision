import numpy as np
import cv2 as cv
# Load a color image in grayscale
img = cv.imread('messi5.jpg',0)

'''
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
'''

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()