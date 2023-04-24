# training leather detection:

# image processing library and hoe to load it because images are basically 3-D arrays 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

# feeding the images to pooling layers
from tensorflow.keras.layers import AveragePooling2D

# automatically dropout certain nodes which creates more bias
from tensorflow.keras.layers import Dropout

# flattening 2x2x3 array
from tensorflow.keras.layers import Flatten

# layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# preprocessing the input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# settig the things to category like 0 and 1
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

# splitting train and test elements
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# paths is a part of paths which rotates and resize the images
from imutils import paths

# basic libraries
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the variables for severa purposes

init_lr = 1e-4
epochs = 2
bs = 32
directory = 'Leather Defect Classification'
