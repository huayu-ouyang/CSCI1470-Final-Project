import tensorflow as tf
import keras
import math
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.callbacks import History
from skimage.transform import resize
import numpy as np
import json
import pydicom
import matplotlib.pyplot as plt
import csv
import random

NUM_IMAGES = 10000
TEST_SIZE = 0.3

BATCH_SIZE = 32 # 64
EPOCHS = 5
IMG_DIM = 256

TRAIN_SIZE = math.floor(NUM_IMAGES * (1 - TEST_SIZE))
TEST_SIZE = math.ceil(NUM_IMAGES * TEST_SIZE)

# .........
train_img_p = 'rsna-pneumonia-detection-challenge/stage_2_train_images/'
test_imgp = 'rsna-pneumonia-detection-challenge/stage_2_test_images/'
train_csv = 'rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'

all_files = []
all_labels = []
all_images = []


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
        else:
            image_arr = dcm_data.pixel_array
            resized_arr = resize(image_arr, (IMG_DIM,IMG_DIM))

            # if ct == 1:
            #     plt.imsave("nopneumonia.png", resized_arr)
            # if ct == 4:
            #     plt.imsave("pneumonia.png", resized_arr)

            all_images.append(resized_arr.flatten())
            all_files.append(fn)
            all_labels.append(label)
            # print(label)
        # else:
        #     image_arr = dcm_data.pixel_array
        #     resized_arr = resize(image_arr, (IMG_DIM,IMG_DIM))

        #     """
        #     plt.imshow(resized_arr)
        #     plt.show()
        #     plt.clf()

        #     """

        #     test_images.append(resized_arr.flatten())
        #     test_files.append(fn)
        #     test_labels.append(label)
            # print(label)
        ct += 1

all_files = np.array(all_files)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

c = list(zip(all_labels, all_images))
random.seed(4)
random.shuffle(c)

all_labels, all_images = zip(*c)
all_labels = np.array(list(all_labels))
all_images = np.array(list(all_images))

train_labels = all_labels[:TRAIN_SIZE]
train_images = all_images[:TRAIN_SIZE]



# test_files = np.array(test_files)
test_labels = all_labels[TRAIN_SIZE:]
test_images = all_images[TRAIN_SIZE:]
print(len(test_images))

train_images = np.transpose(\
		np.reshape(train_images,(-1,1,IMG_DIM,IMG_DIM)),[0,2,3,1])

test_images = np.transpose(\
        np.reshape(test_images,(-1,1,IMG_DIM,IMG_DIM)),[0,2,3,1])

print("Test and train transposed")
# Has 1 ouptut channel



# Generators for images
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=60,
    horizontal_flip=True,)

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    horizontal_flip=True,)

f=train_image_generator.flow(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    shuffle=True)

t=test_image_generator.flow(
    test_images,
    test_labels,
    batch_size=BATCH_SIZE)
print("Generators created.")

# View images
# for train_b, labels_b in f:
#     for i in range(0,5):
#         print("IS: ", labels_b[i])
#         plt.imshow(train_b[i].reshape(1024, 1024))
#         plt.show()
#         plt.clf()
#     break
# exit(0)

model = Sequential()
model.add(Conv2D(32, 3, padding='same', input_shape=(IMG_DIM, IMG_DIM ,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))



# model.add(Conv2D(20, 3, padding='same'))
# model.add(BatchNormalization())
# model.add(LeakyReLU())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(20, 3, padding='same'))
# model.add(BatchNormalization())
# model.add(LeakyReLU())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

model.add(Flatten())
# model.add(Conv2D(1, 1, padding='same', activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
# print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy",metrics=['accuracy'])
# print("model created.")

# H = History()
H = model.fit_generator(
    f,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE)+1,
    epochs=EPOCHS,
    validation_data=t,
    validation_steps=(TEST_SIZE // BATCH_SIZE)+1)


print(H.history)
print("model fitted.")

# P = model.predict_generator(t)
# print(P)

# output = model.evaluate_generator(t)
# print(output)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="test loss")
plt.xlabel("Epoch #")

plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("cnnloss3.png")

plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history['accuracy'], label="train accuracy")
plt.plot(np.arange(0, EPOCHS), H.history['val_accuracy'], label="test accuracy")
plt.xlabel("Epoch #")

plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig("cnnaccuracy3.png")
