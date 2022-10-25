import tensorflow as tf
import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio


import datasets
import utils
import models
tf.compat.v1.enable_eager_execution()
import sys

print('sys.argv[1]={}'.format(sys.argv[1]))
object_name = sys.argv[1]

PROJECT_DIR = "./"


VISAPP_DATA_DIR = 'dataDir/' + object_name
VISAPP_MODEL_FILE = PROJECT_DIR + 'model/' + object_name + '_model.h5'
VISAPP_TEST_SAVE_DIR = 'test/' + object_name


BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = int(sys.argv[2])

dataset = datasets.Visapp(VISAPP_DATA_DIR, 'filename_list.txt', object_name, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.7)

net = models.AlexNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net.train(VISAPP_MODEL_FILE, max_epoches=EPOCHS, load_weight=False)

net.test(VISAPP_TEST_SAVE_DIR)
