# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:50:27 2019

@author: vmarimut
"""


import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from keras.layers import  Convolution1D, MaxPooling1D, GlobalMaxPool1D
from keras.layers import LeakyReLU
from keras.models import Sequential

from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import random

train = '../../Input/train'
test = '../../Input/test'


input_length = 16000*5
m=220
K=60
b1=0.5
b2=0.999
batch_size = 32 
prob=0.5
epoch =10
dropout = Dropout(rate=prob)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def train_generator(list_files, batch_size=batch_size):
    while True:
        random.shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):            
            batch_data = [loadData(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, :,np.newaxis]
            batch_labels = [file_to_int[os.path.basename(fpath)] for fpath in batch_files]
            batch_labels = np.array(batch_labels)            
            yield batch_data, batch_labels
            


    
def loadData(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
#    if len(data)>input_length:    
#        max_offset = len(data)-input_length        
#        offset = np.random.randint(max_offset)        
#        data = data[offset:(input_length+offset)]
#    else:
#        if input_length > len(data):
#            max_offset = input_length - len(data)
#            offset = np.random.randint(max_offset)
#        else:
#            offset = 0
#        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
#    data = GetLogMelSpec(data)
    #display_spectogram(data)
    #print(int_to_label[file_to_int[os.path.basename(file_path)]])

    return data


def GetConvRBMModel(X):


    inp = Input(shape=X.shape)
    rbm_model = Sequential()
    rbm_model.add(Convolution1D(K,132,activation=activations.linear,padding='same',input_shape=X.shape))
    rbm_model.add(LeakyReLU(alpha=0.1))
    dropout.rate = prob
    rbm_model.add(dropout)
    rbm_model.add(MaxPooling1D(50,10))
    model = rbm_model.Model(inputs=inp, outputs=inp)  
    
    opt = optimizers.Adam(beta_1=b1,beta_2=b2)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def GetConvModel():

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


train_files = glob.glob("../../Input/train/*.wav")
test_files = glob.glob("../../Input/test/*.wav")
train_labels = pd.read_csv("../../Input/train.csv")
labels = dict()

#for name,label  in zip(train_labCoels.fname.values,train_labels.label.values):
#    labels[name]=label

for name,label  in zip(train_labels.fname.values,train_labels.label.values):
    labels[name]=label

list_labels = sorted(list(set(train_labels.label.values)))
label_to_int = {k:v for v,k in enumerate(list_labels)}
int_to_label = {v:k for k,v in label_to_int.items()}
file_to_int = {k:label_to_int[v] for k,v in labels.items()}

tr_files, val_files = train_test_split(sorted(train_files), test_size=0.1, random_state=42)

model = GetConvRBMModel()
#model.load_weights("baseline_cnn_gammatone.h5")
model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=epoch,
                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,
                   use_multiprocessing=False, workers=8, max_queue_size=60,
                    callbacks=[ModelCheckpoint("baseline_cnn_gammatone.h5", monitor="val_acc", save_best_only=True),
                               EarlyStopping(patience=5, monitor="val_acc")])
    
model.save_weights("baseline_cnn_gammatone.h5")
    