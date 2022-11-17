import sys

import cv2  
import numpy as np

sys.setrecursionlimit(10 ** 9)

import drag_and_move_rectangle as dmr

wName = "Annotation for Object Detection"
imageWidth 	= 320
imageHeight = 240
image = np.ones([imageHeight, imageWidth, 3], dtype=np.uint8)  
image *= 255

rectI = dmr.DragRectangle(image, wName, imageWidth, imageHeight)

cv2.namedWindow(rectI.wname)
cv2.setMouseCallback(rectI.wname, dmr.dragrect, rectI)


while True:
    cv2.imshow(wName, rectI.image)
    key = cv2.waitKey(1) & 0xFF


    if rectI.returnflag:
        break

print("Dragged rectangle coordinates")
print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
      str(rectI.outRect.w) + ',' + str(rectI.outRect.h))

# close all open windows
cv2.destroyAllWindows()
