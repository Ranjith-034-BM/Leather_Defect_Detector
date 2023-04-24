# leather defect classifier

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
import pandas as pd
import tensorflow as tk
import keras as ke
import sklearn
import cv2
import PIL

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , AveragePooling2D ,Dense
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# importing the datasets

# directory

DIR = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification"
x =[]
y =[]

pinhole = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\pinhole"
loose_grain = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\loose grains"
growth_marks = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\Growth marks"
folding_marks = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\Folding marks"
grain_off = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\Grain off"
# non_defective = "D:\Python Projects\Leather defect detection\dataset\Leather Defect Classification\non defective"



def assign_type(img , leather_type):
    return assign_type

def make_training_data(leather_type , DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR , img)
        label = assign_type(img , leather_type)
        img = cv2.imread(path , cv2.IMREAD_COLOR)
        img = cv2.resize(img , (300,150))

        x.append(img)
        y.append(str(label))

# making the dataset

make_training_data("pinhole",pinhole)
print(len(x))

make_training_data("grain off" , grain_off)
print(len(x))

make_training_data("folding marks",folding_marks)
print(len(x))

make_training_data("loose grains",loose_grain)
print(len(x))

make_training_data("growth_marks",growth_marks)
print(len(x))

'''
make_training_data("non defective",non_defective)
print(len(x))
'''
# storing the data

# categorizing the label

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y,6)

# making the image pixels between 0 and 1

x = np.array(x)
x = x/255

# seperating the data

x_train,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print(x_train.shape,x_test.shape)

# model training

model = Sequential()

# first stage
model.add(Conv2D(filters = 32 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# secoind stage
model.add(Conv2D(filters = 64 , kernel_size = (3,3)))
model.add(AveragePooling2D(pool_size = (2,2) , strides = (1,1)))

# third stage
model.add(Conv2D(filters = 64 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# fourth stage
model.add(Conv2D(filters = 128 , kernel_size = (3,3)))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (1,1)))

# flattening layer

model.add(Flatten())

# input layer

model.add(Dense(6,activation = "relu",input_shape = (300,150)))

# hidden layers

model.add(Dense(12,activation = "relu"))

# output layer

model.add(Dense(6 , activation = "softmax"))

history = model.compile(loss = "categorical_crossentropy" , optimizer = Adam(lr = 0.001) , metrics = ['accuracy'])

print(history)

from keras.callbacks import ReduceLROnPlateau
red_lr = ReduceLROnPlateau(monitor = 'val_acc',patience = 3,factor = 0.1,verbode = 1)

datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True)

batch_size = 128
epochs = 10

model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = epochs,validation_data = (x_test,y_test),verbose = 1,
                    steps_per_epoch = x_train.shape[0] // batch_size)

result = model.evaluate(x_test,y_test,batch_size = 128)
print("loss and accuracy: ",result)

prediction = model.predict(x_test)
print(prediction[2])

plt.figure()
plt.imshow(x_test[2])
plt.colormap()
plt.show()

import cv2
import numpy as np

# Load the image
img = cv2.imread(x_test[2])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Calculate the surface area of the largest contour
max_contour = max(contours, key=cv2.contourArea)
area = cv2.contourArea(max_contour)

# Display the image with the contours and area
cv2.imshow("Image with contours", img)
print("Surface area of object in image is", area)

cv2.waitKey(0)
cv2.destroyAllWindows()

# pixel approach

pixel_1_cm_value = 0.1

from PIL import Image

image = Image.open(x_test[2])
width , height = image.size

number_of_pixels = width*height

surface_area = pixel_1_cm_value*number_of_pixels
print("surface area of the image is {0} cm^2".format(number_of_pixels))

# findimg the contour size

print(contours)
width , height , z = contour.size
contour_no_of_pixels = width*height*z

surface_area_of_contour = pixel_1_cm_value*contour_no_of_pixels

print("surface are of the contour is {0} cm^2".format(surface_area_of_contour))
