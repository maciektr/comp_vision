import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import random

from scipy.spatial.transform import Rotation as Rot

def split_samples(samples_file, train_file, test_file, ratio=0.8):
    with open(samples_file) as samples_fp:
        lines = samples_fp.readlines()
        random.shuffle(lines)

        train_num = int(len(lines) * ratio)
        test_num = len(lines) - train_num
        count = 0
        data = []
        for line in lines:
            count += 1
            data.append(line)
            if count == train_num:
                with open(train_file, "w+") as train_fp:
                    for d in data:
                        train_fp.write(d)
                data = []

            if count == train_num + test_num:
                with open(test_file, "w+") as test_fp:
                    for d in data:
                        test_fp.write(d)
                data = []
    return train_num, test_num
                
def get_list_from_filenames(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class Visapp:
    def __init__(self, data_dir, data_file,object_name, batch_size=64, input_size=64, ratio=0.8):
        self.object_name = object_name
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.train_file = None
        self.test_file = None
        self.poses = {};
        self.__gen_filename_list(os.path.join(self.data_dir, self.data_file))
        self.train_num, self.test_num = self.__gen_train_test_file(os.path.join(self.data_dir, 'train.txt'),
                                                                   os.path.join(self.data_dir, 'test.txt'), ratio=ratio)
        self.initPoses()

    def initPoses(self):
        poses_path = os.path.join(self.data_dir, self.object_name + "_pose.txt")
        posesFile = open(poses_path, 'r')
        for line in posesFile.readlines():
            splited = line.split("\n")[0].split(' ')
            tab = []
            print(splited)
            for r in splited[1:]:
                tab.append(float(r))
            self.poses[splited[0]] = tab
        posesFile.close()


    def get_input_img(self, data_dir, file_name, img_ext='.png', annot_ext='.txt'):
        img = cv2.imread(os.path.join(data_dir, file_name + img_ext))
        x_min, x_max, y_min, y_max = utils.getBbox(os.path.join(data_dir, file_name + img_ext))
        k = 0.3
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        crop_img = img[int(y_min): int(y_max), int(x_min): int(x_max)]

        crop_img = cv2.resize(crop_img, (self.input_size, self.input_size))
        
        crop_img = np.asarray(crop_img)
        normed_img = (crop_img - crop_img.mean())/crop_img.std()
        
        return normed_img
        
    
    def __get_input_label(self, data_dir, file_name, annot_ext='.txt'):
        R = self.poses[file_name.split('/')[1] + ".png"]
        roll, yaw, pitch =   utils.euler_from_quaternion(R[0],R[1],R[2],R[3])

        # Bin values
        bins = np.array(range(-99, 99, 3))
        bin_labels = np.digitize([yaw, pitch, roll], bins) - 1
    
        cont_labels = [roll, yaw, pitch]
    
        return bin_labels, cont_labels

    def __gen_filename_list(self, filename_list_file):
        if not os.path.exists(filename_list_file):
            with open(filename_list_file, 'w+') as tlf:
                for root, dirs, files in os.walk(self.data_dir):
                    for subdir in dirs:
                        subfiles = os.listdir(os.path.join(self.data_dir, subdir))
                    
                        for f in subfiles:
                            if os.path.splitext(f)[1] == '.png':
                                token = os.path.splitext(f)[0].split('-')
                                filename = os.path.splitext(f)[0]
                                # print(filename)
                                tlf.write(subdir + '/' + filename + '\n')
    
    def __gen_train_test_file(self, train_file, test_file, ratio=0.5):
        self.train_file = train_file
        self.test_file = test_file
        return split_samples(os.path.join(self.data_dir, self.data_file), self.train_file, self.test_file, ratio=ratio)
    
    def train_num(self):
        return self.train_num
    
    def test_num(self):
        return self.test_num
    
    def save_test(self, name, save_dir, prediction):
        img_path = os.path.join(self.data_dir, name + '.png')
        print(prediction)
        cv2_img = cv2.imread(img_path)
        cv2_img = utils.draw_axis(cv2_img, prediction[0], prediction[1], prediction[2], tdx=200, tdy=200,
                            size=100)
        save_path = os.path.join(save_dir, name.split('/')[1] + '.png')
        # print(save_path)
        cv2.imwrite(save_path, cv2_img)
    def save_test_real(self, name, save_dir, prediction):
        img_path = os.path.join(self.data_dir, name + '.png')
        print(prediction)
        cv2_img = cv2.imread(img_path)
        cv2_img = utils.draw_axis(cv2_img, prediction[0], prediction[1], prediction[2], tdx=200, tdy=200,
                            size=100)
        save_path = os.path.join(save_dir, name.split('/')[1] + '_real.png')
        # print(save_path)
        cv2.imwrite(save_path, cv2_img)
    def testData(self):
        sample_file = self.test_file
        # if test:
        #     sample_file = self.test_file

        filenames = get_list_from_filenames(sample_file)
        res = []
        for name in filenames:
            img = self.get_input_img(self.data_dir, name)
            res.append((img, name))
        return res

    def data_generator(self, shuffle=True, test=False):
        sample_file = self.train_file
        if test:
            sample_file = self.test_file
    
        filenames = get_list_from_filenames(sample_file)
        file_num = len(filenames)
        
        while True:
            if shuffle and not test:
                idx = np.random.permutation(range(file_num))
                filenames = np.array(filenames)[idx]
            max_num = file_num - (file_num % self.batch_size)
            for i in range(0, max_num, self.batch_size):
                batch_x = []
                batch_yaw = []
                batch_pitch = []
                batch_roll = []
                names = []
                my_batch = []
                for j in range(self.batch_size):
                    img = self.get_input_img(self.data_dir, filenames[i + j])
                    bin_labels, cont_labels = self.__get_input_label(self.data_dir, filenames[i + j])
                    #print(img.shape)
                    # my_batch .append([bin_labels[3], cont_labels[3]])
                    batch_x.append(img)
                    batch_yaw.append([bin_labels[0], cont_labels[0]])
                    batch_pitch.append([bin_labels[1], cont_labels[1]])
                    batch_roll.append([bin_labels[2], cont_labels[2]])
                    names.append(filenames[i + j])

                my_batch = np.array(my_batch)
                batch_x = np.array(batch_x, dtype=np.float32)
                batch_yaw = np.array(batch_yaw)
                batch_pitch = np.array(batch_pitch)
                batch_roll = np.array(batch_roll)
                
                if test:
                    yield (batch_x, [batch_yaw, batch_pitch, batch_roll], names)
                else:
                    yield (batch_x, [batch_yaw, batch_pitch, batch_roll])
            if test:
                break

