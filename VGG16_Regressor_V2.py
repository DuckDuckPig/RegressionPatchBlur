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
import models

## Parameters ##
parent_dir = '/home/varelal/Documents/COCO_blurred_V1/'

train_dir = parent_dir + 'Train/'
test_dir = parent_dir + 'Test/'
val_dir = parent_dir + 'Validate/'

train_labels = parent_dir + 'train_dataset.csv'
test_labels = parent_dir + 'test_dataset.csv'
val_labels = parent_dir +'val_dataset.csv'

csv_logger_dir = 'log/VGG14.csv'
weights_dir = 'Wtrim/VGG14.h5'

epoch = 0
PATCH_SIZE = [29,30,31,32,33] #[30, 32, 48, 30, 32, 64, 30, 32, 48, 64]
CHANNELS = 3
TRAIN_BATCH_SIZE = 50
VAL_BATCH_SIZE = 32
EPOCHS = 200

#Filtering
max_patch = 33

## Functions 
def cot(x):
    return 1/np.tan(x)

def patch_filter(df, patch):
    test_f = pd.DataFrame(columns=df.columns.tolist())
    
    for theta in range(-90,90):
        theta_r = theta * np.pi / 180
        test_t = df[df['angle'] == theta]
    
        if abs(theta) <= 45:
            test_f = pd.concat([test_f,test_t[test_t['length'] < patch*np.sqrt(1+np.tan(abs(theta_r)) ** 2)]], ignore_index=True)
        else:
            test_f = pd.concat([test_f,test_t[test_t['length'] < patch*np.sqrt(1+cot(abs(theta_r)) ** 2)]], ignore_index=True)  

    return test_f
        
        
#Version 2 where we input all dataframe as one
class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, dataframe, epoch, len_y= 2, batch_size=32, n_channels=3, shuffle=True, max_a = 90, min_a = -89, max_l=100, min_l=1):
        self.epoch = -1
        self.flag = -1
        #self.dim = PATCH_SIZE[self.epoch % len(PATCH_SIZE)]  #Will add rotation of patch size
        self.directory = directory
        self.batch_size = batch_size
        #self.y_length = y_length
        #self.y_angle = y_angle
        #self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.len_y = len_y
        
        #Initialize filters
        self.max_a = max_a
        self.min_a = min_a
        self.max_l = max_l
        self.min_l = min_l
        
        self.on_epoch_end()
        
        #print("epoch {}: PATCH_SIZE = {}".format(epoch, self.dim))

    def on_epoch_end(self):
        self.epoch += 1
        
        if self.epoch % 1 == 0:
            self.flag += 1 #increment counter to change patch sizei
            self.dim = PATCH_SIZE[self.flag % len(PATCH_SIZE)]
        
        self.index = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.index)

        print("epoch {}: PATCH_SIZE = {}".format(self.epoch, self.dim))

    def __len__(self):

        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        #Generates one batch of data
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        
        #find list of IDs
        list_IDs_temp = [self.dataframe.loc[k] for k in index]
        
        #enerate data
        X, y = self.__data_generation(list_IDs_temp)
        
        #print("Shape: ", np.unique(X))
        
        return X, y
        
    def random_crop(self, img, random_crop_size):

        height, width = img.shape[0], img.shape[1]      
        dy, dx = random_crop_size
        
        if height >= dy and width >= dx:
            x = np.random.randint(0,width - dx + 1)
            y = np.random.randint(0,height - dy + 1)
        
            return img[y:(y+dy), x:(x+dx),:]/255.0

        else:
            return 0
        
    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples X: (n_samples, *dim, n_channels)
        
        #X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
        #y = np.empty((self.batch_size, self.len_y), dtype=float)
        
        X = []
        y = []
        #print("listID:", type(list_IDs_temp))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print("ID:",ID['filename'], ID['angle'], ID['length'])
            #print("i:",i)
            if os.path.exists(self.directory + ID['filename']):
                #store sample
                img = keras.preprocessing.image.load_img(self.directory + ID['filename'])
                img = keras.preprocessing.image.img_to_array(img)
                height, width = img.shape[0], img.shape[1]                
                dy = self.dim
                dx = self.dim
                
                # Denormalize Length and Angle to filter training
                L = ID['length'] * (self.max_l - self.min_l) + self.min_l
                A = ID['angle'] * (self.max_a - self.min_a) + self.min_a
                Ar = A * np.pi / 180
                
                if height >= dy and width >= dx:
                
                    if (abs(A) <= 45 and L < self.dim * np.sqrt(1+np.tan(abs(Ar))**2)) or (abs(A) > 45 and L < self.dim * np.sqrt(1+cot(abs(Ar))**2)):
                        #X[i,] = self.random_crop(img, random_crop_size=[self.dim, self.dim])
                        X.append(self.random_crop(img, random_crop_size=[self.dim, self.dim]))
                        #store class
                        #y[i,] = [ID['length'],ID['angle']]
                        y.append([ID['length'],ID['angle']])
                    
                else:
                    continue
                    
            else:
                print("Error couldn't load:", self.directory + ID['filename'])
                
        X = np.asarray(X)
        y = np.asarray(y)
        #print(X.shape, y.shape)
        return X, y


## Define model and training

model = models.vgg_14()
model.summary()
# params = {len_y: int(2),
#           dim: (224,224),
#           batch_size: int(32),
#           n_channels: int(3),
#           shuffle: True}

#Filters both datasets to fit a patch max patch
train_pd = pd.read_csv(train_labels)
#train_pd = patch_filter(train_pd, max_patch)
train_pd = train_pd[train_pd['length'] <= max_patch].reset_index()
print("Train:",train_pd.size)

#grab normalized parameters
angle_max = max(train_pd['angle'])
angle_min = min(train_pd['angle'])
length_max = max(train_pd['length'])
length_min = min(train_pd['length'])

train_pd['angle'] = (train_pd['angle'] - min(train_pd['angle'])) / (max(train_pd['angle']) - min(train_pd['angle']))
train_pd['length'] = (train_pd['length'] - min(train_pd['length'])) / (max(train_pd['length']) - min(train_pd['length']))


val_pd = pd.read_csv(val_labels)
#val_pd = patch_filter(val_pd, max_patch)
val_pd = val_pd[val_pd['length'] <= max_patch].reset_index()
print("Val:", val_pd.size)
val_pd['angle'] = (val_pd['angle'] - min(val_pd['angle'])) / (max(val_pd['angle']) - min(val_pd['angle']))
val_pd['length'] = (val_pd['length'] - min(val_pd['length'])) / (max(val_pd['length']) - min(val_pd['length']))

#Generators
"""
train_generator = DataGenerator(train_dir,
                                train_pd['filename'],
                                train_pd['length'],
                                train_pd['angle'])#,
                                #*params)

val_generator = DataGenerator(val_dir,
                              val_pd['filename'],
                              val_pd['length'],
                              val_pd['angle'])#,
                              #*params)
"""
train_generator = DataGenerator(train_dir,
                                train_pd,
                                epoch,
                                batch_size=TRAIN_BATCH_SIZE,
                                max_a = angle_max,
                                min_a = angle_min,
                                max_l = length_max,
                                min_l = length_min)

val_generator = DataGenerator(val_dir,
                              val_pd,
                              epoch,
                              batch_size=VAL_BATCH_SIZE,
                              max_a = angle_max,
                              min_a = angle_min,
                              max_l = length_max,
                              min_l = length_min)
    
#Callbacks
checkpoint = ModelCheckpoint(filepath=weights_dir,
                            save_best_only = True)

earlystop = EarlyStopping(monitor='val_loss',
                         min_delta=10e-12,
                         patience=15,
                         mode='min',
                         restore_best_weights=False)

csv_logger = CSVLogger(csv_logger_dir, append=True, separator=';')

#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#              1e-1,
#              decay_steps = 100000,
#              decay_rate=0.96,
#              staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.10, epsilon=0.1),
              loss=tf.keras.losses.MeanSquaredError())

model.fit(train_generator,
                validation_data = val_generator,
                epochs = EPOCHS,
                callbacks=[checkpoint, earlystop, csv_logger])
