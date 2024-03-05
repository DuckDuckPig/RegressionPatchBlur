import numpy as np
import pandas as pd
import os
import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Reshape, Activation, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda, GlobalAveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def model_v4():
    model = Sequential()

    model.add(Conv2D(input_shape=(None, None, 3), filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid')) 
    
    return model


def model_v5():
    model = Sequential()

    model.add(Conv2D(input_shape=(None, None, 3), filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid')) 
    
    return model
 
def vgg_14():
    model = Sequential()

    model.add(Conv2D(input_shape=(None, None, 3),filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid')) 
    
    return model
    
