from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import gzip
import numpy as np

import json
token = {"username":"jennifer2020","key":"ecdd8b3ac0496be65f1c03e5c389a939"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c rsna-pneumonia-detection-challenge -p /content
!unzip \*.zip
!ls

!pip install pydicom
import pydicom
import pandas as pd
import tensorflow as tf
import gzip
import numpy as np
d = pd.read_csv('/content/stage_2_train_labels.csv')
d.head()
dcm_file = 'fffec09e-8a4a-48b1-b33e-ab4890ccd136.dcm'
dcm_data = pydicom.read_file(dcm_file)

"""
print(dcm_data)
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
import matplotlib.pyplot as plt
plt.imshow(im, cmap=plt.cm.bone)
"""


info = d.to_dict('index')
pnemonia = {}

for key in info:
  pid = info[key]['patientId']
  pnemonia[pid] = info[key]['Target']

limit = 0

small_train_image_arrs = []
train_images = []
small_train_labels = []

image_path_to_type = {}
image_fp = []
image_lb = []
for pid in pnemonia:
  if limit == 2000:
    break
  fn = pid + '.dcm'
  image_path_to_type[fn] = pnemonia[pid]
  image_fp.append(fn)
  dcm_data = pydicom.read_file(fn)

  # train_images.append(dcm_data)
  train_images.append(dcm_data.pixel_array.flatten())
  # small_train_labels.append(pnemonia[pid])
  image_lb.append(pnemonia[pid])
  limit += 1
image_lb = np.array(image_lb)
train_images = np.array(train_images)
print(train_images.shape)
train_images = np.transpose(\
		np.reshape(train_images,(-1, 4, 512 ,512)),[0,2,3,1])
#dd = pd.DataFrame(list(zip(image_fp,image_cls)),columns=['filename','class'])
train_images.shape
# Has 4 ouptut channels



# print("pneumonia path: {}".format(files_path))
# !ls data -l --block-size=M

# !mkdir data/
# !mkdir fp_data/
"""
zip_loc = '/content'
train_dir = loc + '/train'
test_dir = loc + '/test'
"""

BATCH_SIZE = 128
EPOCHS = 10
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
TRAIN_SIZE = 2000
TEST_SIZE = 2000


# Generators for images
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=45,
    horizontal_flip=True)

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    horizontal_flip=True)

# t = np.array(small_train_image_arrs).reshape((128, 1024, 1024))
# train_images = np.zeros((128, 1024, 1024))
# Input data. Numpy array of rank 4 or a tuple.

train_labels = np.zeros((1,100))

train_image_generator.flow(
    train_images,
    image_lb,
    batch_size=32,
    shuffle=True
)

model = tf.keras.Sequential([
  Conv2D(16,kernel_size=(5,5),strides=(2,2), padding='same', activation="relu", use_bias=True, kernel_initializer='glorot_uniform'),
  MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same'),
  Dropout(0.3)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="sparse_categorical_crossentropy")
model.fit_generator(train_image_generator, steps_per_epoch=((train_size // BATCH_SIZE) + 1), epochs=EPOCHS, verbose=2)
