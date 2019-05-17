# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:58:28 2019

@author: vmarimut
"""
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def GetMELModel():

    nclass = len(list_labels)
    inp = Input(shape=(157, 320, 1))
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 7))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model