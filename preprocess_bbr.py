from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import glob
import os
import pydicom as dicom
import cv2

train = pd.read_csv('stage_2_train_labels.csv', engine='python')
train.head()

image_dim = 228

images = []
bboxes = []
classes = []

for _, row in train.iterrows():

    if row.Target == 0:
        continue

    width = row.width
    height = row.height
    cell_type = [0, 1]
    image = "train_imgs/" + row.patientId + ".dcm"
    ds = dicom.dcmread(image)
    pixel_array_np = ds.pixel_array
    image = image.replace(".dcm", ".jpg")
    cv2.imwrite(image, pixel_array_np)
    img_path = "train_imgs/" + row.patientId + ".jpg"
    image = Image.open(img_path).resize((image_dim, image_dim))
    bounding_box = [0.0]*4
    x_min = row.x
    bounding_box[0] = x_min/image_dim
    y_min = row.y
    bounding_box[1] = y_min/image_dim
    x_max = row.x+width
    bounding_box[2] = x_max/image_dim
    y_max = row.y+height
    bounding_box[3] = y_max/image_dim
    classes.append(cell_type)
    bboxes.append(bounding_box)
    images.append(np.asarray(image)/255)


bboxes = np.array(bboxes)
classes = np.array(classes)
X = images
Y = np.concatenate([bboxes, classes], axis=1)

train_X, test_X ,train_Y, test_Y = train_test_split(X, Y, test_size=0.4)

np.save('train_x.npy', train_X)
np.save('train_y.npy', train_Y)
np.save('test_x.npy', test_X)
np.save('test_y.npy', test_Y)
