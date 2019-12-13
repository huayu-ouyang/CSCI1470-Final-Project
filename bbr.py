import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, MaxPooling2D
from PIL import Image, ImageDraw
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 32
epochs = 10
input_dim = 128
num_images = 50
test_size = 0.4
train_size = math.floor(num_images * (1 - test_size))
test_size = math.ceil(num_images * test_size)

train_X = np.load('train_x.npy')
train_Y = np.load('train_y.npy')
test_X = np.load('test_x.npy')
test_Y = np.load('test_y.npy')

train_X = np.reshape(train_X,(-1,1,input_dim,input_dim))
test_X = np.reshape(test_X,(-1,1,input_dim,input_dim))

model = tf.keras.Sequential()
model.add(Conv2D(16, 1, strides=1))
model.add(LeakyReLU())
model.add(Conv2D(16, 1, strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(32, 1, strides=1))
model.add(LeakyReLU())
model.add(Conv2D(32, 1, strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(64, 1, strides=1))
model.add(LeakyReLU())
model.add(Conv2D(64, 1, strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(128, 1, strides=1))
model.add(LeakyReLU())
model.add(Conv2D(128, 1, strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(256, 1, strides=1))
model.add(LeakyReLU())
model.add(Conv2D(256, 1, strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D((1,1)))

model.add(Flatten())

model.add(Dense(1240))
model.add(LeakyReLU())
model.add(Dense(640))
model.add(LeakyReLU())
model.add(Dense(480))
model.add(LeakyReLU())
model.add(Dense(120))
model.add(LeakyReLU())
model.add(Dense(62))
model.add(LeakyReLU())

model.add(Dense(6))
model.add(LeakyReLU())

def calculate_iou(target, prediction):
    xA = tf.keras.backend.maximum(target[...,0],prediction[...,0])
    yA = tf.keras.backend.maximum(target[...,1],prediction[...,1])
    xB = tf.keras.backend.minimum(target[...,2],prediction[...,2])
    yB = tf.keras.backend.minimum(target[...,3],prediction[...,3])
    area = tf.keras.backend.maximum(0.0,xB-xA) * tf.keras.backend.maximum(0.0,yB-yA)
    boxA = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    boxB = (prediction[..., 2] - prediction[..., 0]) * (prediction[..., 3] - prediction[..., 1])
    iou = area / (boxA + boxB - area)
    return iou

def loss_function(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1-iou)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=loss_function,
    metrics=[calculate_iou])

history = model.fit(train_X, train_Y,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(test_X, test_Y))

target_boxes = test_Y * input_dim
predictions = model.predict(test_X)
prediction_boxes = predictions[...,0:4] * input_dim
prediction_classes = predictions[...,4:]
iou_scores = calculate_iou(target_boxes, prediction_boxes)
print("iou score: " + str(iou_scores.numpy().mean()))

for i in range(predictions.shape[0]):
    predicted_b = predictions[i, 0:4] * input_dim
    img = test_X[i] * 255
    img = img.reshape(-1, img.shape[1])
    source_img = Image.fromarray(img.astype(np.uint8), 'L')
    draw = ImageDraw.Draw(source_img)
    # black is predicted
    draw.rectangle(predicted_b, outline=1)
    actual_b = test_Y[i][0:4] * input_dim
    # white is actual
    draw.rectangle([(actual_b[0], actual_b[1]),(actual_b[2], actual_b[3])], outline=300)
    source_img.save('inference_images/image_{}.png'.format(i+1), 'png')

print(history.history.keys())

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="train loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="test loss")
plt.xlabel("Epoch #")

plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("bbrloss.png")

plt.figure()
plt.plot(np.arange(0, epochs), history.history['calculate_iou'], label="train accuracy")
plt.plot(np.arange(0, epochs), history.history['val_calculate_iou'], label="test accuracy")
plt.xlabel("Epoch #")

plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig("bbraccuracy.png")
