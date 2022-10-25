import numpy as np
import cv2 as cv

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

img = cv.imread('messi5.jpg',0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.savefig('foo.png')
plt.savefig('foo.pdf')

plt.show()