import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import pandas as pd
from matplotlib import pyplot as plt

# import tensorflow.contrib.eager as tfe
tf.compat.v1.enable_eager_execution()

# np.set_printoptions(threshold=np.nan)

EPOCHS=25


class IoU(tf.keras.metrics.Metric):                                        #Custom IoU class, inheritance of Metric class
  def __init__(self, **kwargs):
    super(IoU, self).__init__(**kwargs)                                    #super() allow us to use Metric class' arguments

    self.iou = self.add_weight(name='iou', initializer='zeros')            #add_weight: "Adds a new variable to the layer"
    self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
    self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    def get_box(y):                                                        #a function for getting bounding box coordinates
      rows, cols = y[:, 0], y[:, 1]
      rows, cols = rows * 144, cols * 144
      y1, y2 = rows, rows + 52
      x1, x2 = cols, cols + 52
      return x1, y1, x2, y2

    def get_area(x1, y1, x2, y2):                                          #a function for getting area of bounding box
      return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

    gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)                           #getting ground truth bounding box coordinates
    p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)                               #getting prediction bounding box coordinates

    i_x1 = tf.maximum(gt_x1, p_x1)                                         #return the maximum of gt_x1 and p_x1, assign to i_x1
    i_y1 = tf.maximum(gt_y1, p_y1)                                         #return the maximum of gt_x1 and p_x1, assign to i_x1
    i_x2 = tf.minimum(gt_x2, p_x2)                                         #return the maximum of gt_x1 and p_x1, assign to i_x1
    i_y2 = tf.minimum(gt_y2, p_y2)                                         #return the maximum of gt_x1 and p_x1, assign to i_x1

    i_area = get_area(i_x1, i_y1, i_x2, i_y2)                              #area of intersection(or overlap)
    u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area #area of union

    iou = tf.math.divide(i_area, u_area)                                   #calculate the iou
    self.num_ex.assign_add(1)
    self.total_iou.assign_add(tf.reduce_mean(iou))                         #tf.reduce_mean():"Computes the mean of elements across dimensions of a tensor"
    self.iou = tf.math.divide(self.total_iou, self.num_ex)                 #total_iou divide by num_ex, then assign to iou

  def result(self):
    return self.iou

  def reset_state(self):                                                   #reseting the state
    self.iou = self.add_weight(name='iou', initializer='zeros')
    self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
    self.num_ex = self.add_weight(name='num_ex', initializer='zeros')


class AlexNet:
    def __init__(self, dataset, class_num, batch_size, input_size):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
        self.model = self.__create_model()

    def __loss_angle(self, y_true, y_pred, alpha=0.005):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        cls_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf.keras.utils.to_categorical(bin_true,66), logits=y_pred)
        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1)* 3 - 99
        mse_loss = tf.losses.mean_squared_error(cont_true, pred_cont)
        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))

        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)

        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)


        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])

        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle        }

        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
            run_eagerly=True,
            loss=losses,
            metrics=[IoU(name='iou')]
        )

        return model

    @staticmethod
    def plot_learning_curve(history):
        # plt.plot(history.history['loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.show()
        # pd.DataFrame(history.history).plot(figsize=(10,5))
        # plt.show()
        pass



    def train(self, model_path, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()

        if load_weight:
            self.model.load_weights(model_path)
        else:
            history = self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    max_queue_size=10,
                                    workers=1,
                                    verbose=1)

            self.model.save(model_path)
            self.__class__.plot_learning_curve(history)

    def test(self, save_dir):
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            print(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1)* 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1)* 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1)* 3 - 99

            # print(pred_cont_yaw.shape)
            # pred_cont_yaw = predictions[0,:,:]
            # pred_cont_pitch = predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
            # pred_cont_roll = predictions[2,:,:]
            for i in range(len(names)):
                self.dataset.save_test(names[i], save_dir, [pred_cont_yaw[i], pred_cont_pitch[i], pred_cont_roll[i]])
                self.dataset.save_test_real(names[i], save_dir, [batch_yaw[i][0], batch_pitch[i][0], batch_roll[i][0]])


    def test_online(self, image, imageName, save_dir):
        predictions = self.model.predict(image, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        # print(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) *3 - 180
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 180
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 180

        self.dataset.save_test(imageName, save_dir, [pred_cont_yaw, pred_cont_pitch, pred_cont_roll])
