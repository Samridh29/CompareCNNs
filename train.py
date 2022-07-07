import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import alexnet as AlexNet
# import LeNet 
# import VGG
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

EPOCHS = 150
INIT_LR = 1e-3
BS = 32
IMG_DIMS = (96, 96, 3)

data = []
labels = []

imagepaths = sorted(list(paths.list_images("cifar10/train")))
random.seed(42)
random.shuffle(imagepaths)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
for imgs in x_train:
	img = cv2.imread(imgs)
	imgs = cv2.resize(img, (IMG_DIMS[1], IMG_DIMS[0]))
	imgs = img_to_array(imgs)


for imgs in x_test:
	img = cv2.imread(imgs)
	imgs = cv2.resize(img, (IMG_DIMS[1], IMG_DIMS[0]))
	imgs = img_to_array(imgs)


x_train = np.array(x_train, dtype="float") / 255.0
y_train = np.array(y_train)

x_test = np.array(x_test, dtype="float") / 255.0
y_test = np.array(y_test)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.1, height_shift_range = 0.1, 
                        shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")

print("Compiling model...")
alexnet_model = AlexNet.build(width = IMG_DIMS[1], height = IMG_DIMS[0], depth = IMG_DIMS[2], classes = len(set(lb.classes_)))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
alexnet_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = alexnet_model.fit(
	x=aug.flow(x_train, y_train, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(x_train) // BS,
	            epochs=EPOCHS, verbose=1)

print("serializing network...")
alexnet_model.save("alexnet.model", save_format="h5")

print("serializing label binarizer...")
f = open("lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
# plt.savefig(args["plot"])