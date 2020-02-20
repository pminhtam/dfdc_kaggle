import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.05
# config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))
import random
from os import listdir
from os.path import isfile, join

import numpy as np

import json
import matplotlib.pyplot as plt
import cv2
margin = 0.2
from tqdm import tqdm
from mtcnn import MTCNN
import pickle

detector = MTCNN()


# path10 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_10"
# path11 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_11"
# path12 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_12"
# path13 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_13"
# path14 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_14"
# path15 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_15"


# path21 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_21"
# path22 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_22"
# path23 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_23"
# path24 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_24"
# path25 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_25"


# path26 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_26"
# path27 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_27"
# path28 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_28"
# path29 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_29"
# path30 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_30"

# path31 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_31"
# path32 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_32"
# path33 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_33"
# path34 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_34"
# path35 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_35"

# path41 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_41"
# path42 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_42"
# path43 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_43"
# path44 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_44"
# path45 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_45"
# path46 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_46"
path47 = "/hdd/tam/kaggle/www.kaggle.com/c/16880/datadownload/dfdc_train_part_47"


# paths = [path21,path22,path23,path24,path25]
# paths = [path26,path27,path28,path29,path30]
# paths = [path31,path32,path33,path34,path35]
paths = [path47]

IMGWIDTH = 128
margin = 0.2

# save_interval = 6 # perform face detection every {save_interval} frames
for path in paths:
    data = json.load(open(join(path, "metadata.json")))
    print(path)
    for vi in tqdm(data): 
#         if data[vi]['label'] == "REAL":
#             continue
        if data[vi]['label'] == "FAKE":
            if os.path.exists(join("/hdd/tam/kaggle/train_videos/df", vi +".pkl")):
                continue
        if data[vi]['label'] == 'REAL':
            if os.path.exists(join("/hdd/tam/kaggle/train_videos/real", vi+".pkl")):
                continue
            
            
        video = cv2.VideoCapture(join(path, vi))
        success = True
    #     success, vframe = video.read()
        data_videos = []
        save_interval = 13

        success, image = video.read()
        while success:
    #         for i in range(0,video.__len__(),save_interval):
            for i in range(save_interval):
                success, image = video.read()
                if not success:
                    break
    #         print(image.shape)
    #         if image.all() ==None:
    #             continue
            try:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            except:
                continue
        #         face_positions = face_recognition.face_locations(img)
            face_positions = detector.detect_faces(image)
            if len(face_positions) == 0:
                continue
            face_position =face_positions[0]['box']
            x,y,w,h = face_position
            offsetx = round(margin * (w))
            offsety = round(margin * (h))
            y0 = round(max(y - offsety, 0))
            x1 = round(min(x + w + offsetx, image.shape[1]))
            y1 = round(min(y+ h + offsety, image.shape[0]))
            x0 = round(max(x - offsetx, 0))
    #         print(x0,x1,y0,y1)
            face = image[y0:y1,x0:x1]


            face = cv2.resize(face,(IMGWIDTH,IMGWIDTH))
    #         plt.imshow(face)
    #         plt.show()
            data_videos.append(face)
            success, image = video.read()

        data_videos = np.array(data_videos)
        if data[vi]['label'] == "FAKE":
            output = open(join("/hdd/tam/kaggle/train_videos/df", vi +".pkl"),'wb')
            pickle.dump(data_videos, output)
            output.close()
        if data[vi]['label'] == 'REAL':
            output = open(join("/hdd/tam/kaggle/train_videos/real", vi+".pkl"),'wb')
            pickle.dump(data_videos, output)
            output.close()
