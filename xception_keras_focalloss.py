import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))

from keras.preprocessing import image

import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import Counter
from sklearn.utils import class_weight

import glob

for i in glob.glob("./log_xception_keras_focalloss/event*"):
    os.remove(i)
IMAGE_SHAPE = 256
model = keras.models.Sequential([
#     keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(128, 128, 3)),
    # keras.applications.nasnet.NASNetLarge(include_top=False, weights='imagenet',input_shape=(128, 128, 3)),

    keras.applications.Xception(include_top=False, weights='imagenet',input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3)),

    keras.layers.Flatten(),
#     keras.layers.Dense(128,activation = 'relu',kernel_initializer='random_normal'),
#     keras.layers.Dense(16,activation = 'relu',kernel_initializer='random_normal'),
    keras.layers.Dense(1,activation = 'sigmoid')
])


batch_size = 32
dataGenerator = ImageDataGenerator(rescale=1./255,rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,shear_range=0.05)

generator = dataGenerator.flow_from_directory(
#         '/data/tam/kaggle/train_imgs/',
         '/data/tam/kaggle/extract_raw_img/',
        target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',shuffle=True)
 
test_generator = dataGenerator.flow_from_directory(
#         '/data/tam/kaggle/test_imgs/',
    '/data/tam/kaggle/extract_raw_img_test/',
        target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')


from focal_loss import BinaryFocalLoss

model.compile(optimizer = "adam", loss = BinaryFocalLoss(gamma=2),metrics = ['accuracy'])


counter = Counter(generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     

class_weights_2 = class_weight.compute_class_weight(
               'balanced',
                np.unique(generator.classes), 
                generator.classes)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log_xception_keras_focalloss")
checkpoints = keras.callbacks.ModelCheckpoint("./log_xception_keras_focalloss/checkpoint_newdata_{epoch:04d}.pth", monitor='val_loss', verbose=0, save_best_only=False, period=3)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0)

model.fit_generator(generator,validation_data=test_generator,steps_per_epoch=int(1142792/batch_size), epochs=100,workers=4,validation_steps=14298/batch_size,class_weight = class_weights,callbacks = [tensorboard_callback,checkpoints])
