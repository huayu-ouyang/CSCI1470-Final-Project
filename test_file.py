import tensorflow as tf
import keras
import math
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import History
import numpy as np
import json
import pydicom
import matplotlib.pyplot as plt
import csv

NUM_IMAGES = 1024
TEST_SIZE = 0.3
# .........
train_img_p = 'rsna-pneumonia-detection-challenge/stage_2_train_images/'
test_imgp = 'rsna-pneumonia-detection-challenge/stage_2_test_images/'
train_csv = 'rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'

train_files = []
train_labels = []
train_images = []

test_files = []
test_labels = []
test_images = []

with open(train_csv, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    ct = 0
    for row in reader:
        fn = train_img_p + row[0] + '.dcm'
        dcm_data = pydicom.read_file(fn)
        label = int(row[5])
        if ct == NUM_IMAGES:
            break
        elif ct < (NUM_IMAGES / 2):
            train_images.append(dcm_data.pixel_array.flatten())
            train_files.append(fn)
            train_labels.append(label)
        else:
            test_images.append(dcm_data.pixel_array.flatten())
            test_files.append(fn)
            test_labels.append(label)
        ct += 1

train_files = np.array(train_files)
train_labels = np.array(train_labels)
train_images = np.array(train_images)

test_files = np.array(test_files)
test_labels = np.array(test_labels)
test_images = np.array(test_images)

train_images = np.transpose(\
		np.reshape(train_images,(-1,1,1024,1024)),[0,2,3,1])

test_images = np.transpose(\
        np.reshape(test_images,(-1,1,1024,1024)),[0,2,3,1])

print("Test and train transposed")
# Has 1 ouptut channel

BATCH_SIZE = 32 # 64
EPOCHS = 10
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
TRAIN_SIZE = math.floor(NUM_IMAGES * (1 - TEST_SIZE))
TEST_SIZE = math.ceil(NUM_IMAGES * TEST_SIZE)

# Generators for images
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=60,
    horizontal_flip=True)

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    horizontal_flip=True)

f=train_image_generator.flow(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    shuffle=True)
t=train_image_generator.flow(
    test_images,
    test_labels,
    batch_size=BATCH_SIZE)
print("Generators created.")

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

model = Sequential()
model.add(Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)))
model.add(Conv2D(4, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(20, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
# model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy",metrics=['accuracy'])
print("model created.")

# H = History()
H = model.fit_generator(
    f,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE)+1,
    epochs=EPOCHS,
    validation_data=t,
    validation_steps=(TEST_SIZE // BATCH_SIZE)+1)
    # callbacks=[H])
print("model fitted.")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="test loss")
plt.plot(np.arange(0, EPOCHS), H.history['accuracy'], label="train accuracy")
plt.plot(np.arange(0, EPOCHS), H.history['val_accuracy'], label="test accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("summary.png")

