import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin


from skimage import img_as_bool, io, color
from skimage import img_as_uint
import cv2, numpy as np
import math
import tensorflow as tf

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)
	
def getBbox(imageName):


    import numpy as np
    mask = io.imread(imageName)
    mask[mask > 10] = 255
    mask[mask <= 10] = 0

    cv2.boundingRect(mask[mask == 255])
    image = mask
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        xmin = x
        xmax = x + w
        ymin = y
        ymax = y + h
        return (xmin, xmax, ymin, ymax)
		
def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the object.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        object_p_x = tdx - 0.50 * size
        object_p_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        object_p_x = width / 2 - 0.5 * size
        object_p_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + object_p_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + object_p_y
    x2 = size * (-cos(y) * sin(r)) + object_p_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + object_p_y
    x3 = size * (sin(y)) + object_p_x
    y3 = size * (-cos(y) * sin(p)) + object_p_y

    # Draw base in red
    cv2.line(img, (int(object_p_x), int(object_p_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(object_p_x), int(object_p_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-object_p_x),int(y2+y1-object_p_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-object_p_x),int(y1+y2-object_p_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(object_p_x), int(object_p_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-object_p_x),int(y1+y3-object_p_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-object_p_x),int(y2+y3-object_p_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-object_p_x),int(y2+y1-object_p_y)), (int(x3+x1+x2-2*object_p_x),int(y3+y2+y1-2*object_p_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-object_p_x),int(y3+y1-object_p_y)), (int(x3+x1+x2-2*object_p_x),int(y3+y2+y1-2*object_p_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-object_p_x),int(y2+y3-object_p_y)), (int(x3+x1+x2-2*object_p_x),int(y3+y2+y1-2*object_p_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-object_p_x),int(y3+y1-object_p_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-object_p_x),int(y3+y2-object_p_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
