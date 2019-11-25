from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


import gzip
import numpy as np
import json
import pydicom
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

train_img_p = 'rsna-pneumonia-detection-challenge/stage_2_train_images/'
test_imgp = 'rsna-pneumonia-detection-challenge/stage_2_test_images/'
train_csv = 'rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'

train_files = []
train_labels = []
train_images = []

with open(train_csv, 'r', newline='') as f:
    reader = csv.reader(f)

    next(reader)
    ct = 0
    for row in reader:
        if ct == 500:
            break
        fn = train_img_p + row[0] + '.dcm'
        dcm_data = pydicom.read_file(fn)
        label = int(row[5])
        train_images.append(dcm_data.pixel_array.flatten())
        train_files.append(fn)
        train_labels.append(label)
        ct += 1

train_files = np.array(train_files)
train_labels = np.array(train_labels)
train_images = np.array(train_images)


train_images = np.transpose(\
		np.reshape(train_images,(-1,1,1024,1024)),[0,2,3,1])
print(train_images.shape)

# Has 1 ouptut channel

BATCH_SIZE = 32 # 64
EPOCHS = 5 # 10
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
TRAIN_SIZE = 500
TEST_SIZE = 500


# Generators for images
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=60,
    horizontal_flip=True)

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    horizontal_flip=True)
# train_image_generator.fit(train_images)
f=train_image_generator.flow(
    train_images,
    train_labels,
    batch_size=32,
    shuffle=True)

"""
# View images
for train_b, labels_b in f:
    for i in range(0,12):
        print("IS: ", labels_b[i])
        plt.imshow(train_b[i].reshape(1024, 1024))
        plt.show()
        plt.clf()
    break
exit(0)
"""

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
    MaxPooling2D(),
    Conv2D(20, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    # Conv2D(20, 3, padding='same', activation='relu'),
    # MaxPooling2D(),
    Flatten(),
    # Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy",metrics=['accuracy'])
model.fit_generator(f, steps_per_epoch=((TRAIN_SIZE // BATCH_SIZE) + 1), epochs=EPOCHS)
