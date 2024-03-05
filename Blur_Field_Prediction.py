import numpy as np
import pandas as pd
import os
import progressbar
from tqdm import tqdm
import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Reshape, Activation, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda, GlobalAveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, Model
import glob
import models

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def blur_field_gen(M,N,row,col,values):
    """
    M,N - Shape of overall blur field M-Rows, N-Columns
    row,col - is how many row and column regions to shop the blur field into
    values - a 1-Dimentional array values that scans row by row.
    """

    blur_field = np.zeros((M,N))
    M_step = M//row
    N_step = N//col

    index = 0
    for i in range(0,row):
        for j in range(0,col):
            #print(i*M//row,(i+1)*M//row,j*N//col,(j+1)*N//col, index)
            blur_field[i*M//row:(i+1)*M//row,j*N//col:(j+1)*N//col] = values[index]

            if index < len(values)-1:
                index += 1

    return blur_field.astype('int')

PATCH_SIZE = 33
weights_dir = 'Weights_V22'
weight_filename = 'Weights/' + weights_dir + '.h5'
out_dir = 'Prediction_map_test/'

model = models.vgg_14()
model.load_weights(weight_filename)

# Normalized parameters
length_max = 33
length_min = 1
angle_max = 90
angle_min = -89

filenames = sorted(glob.glob('TestPattern_Blurs/*'))

for filename in filenames:
    print(filename)
    img = keras.preprocessing.image.load_img(filename)
    img = keras.preprocessing.image.img_to_array(img)/255.0

    height, width = img.shape[0], img.shape[1]

    print(height,width)

    Lmap_GT = np.zeros((height-PATCH_SIZE,width-PATCH_SIZE))
    Amap_GT = np.zeros((height-PATCH_SIZE,width-PATCH_SIZE))
    Lmap_P = np.zeros((height-PATCH_SIZE,width-PATCH_SIZE))
    Amap_P = np.zeros((height-PATCH_SIZE,width-PATCH_SIZE))


    for x in tqdm(range(0,height-PATCH_SIZE)):
        for y in tqdm(range(0, width-PATCH_SIZE), leave=False):
            patch = np.expand_dims(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:],0)

            pred = model.predict(patch)
            length = (length_max - length_min)*pred[0,0] + length_min
            angle = (angle_max - angle_min)*pred[0,1] + angle_min

            Lmap_P[x,y] = length
            Amap_P[x,y] = angle
    
    outA = out_dir + 'Amap_'+os.path.basename(filename).split('.')[0]+'.csv'
    outL = out_dir + 'Lmap_'+os.path.basename(filename).split('.')[0]+'.csv'
    np.savetxt(outA, Amap_P, delimiter=',')
    np.savetxt(outL, Lmap_P, delimiter=',')
        
    
