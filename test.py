# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 04:56:32 2019

@author: vmarimut
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:55:36 2019

@author: vmarimut
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:36:44 2019

@author: vmarimut
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:38:28 2019

@author: vmarimut
"""

import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPool2D, GlobalMaxPool2D,Activation,Flatten,concatenate
from sklearn.model_selection import train_test_split
from keras.utils import Sequence, to_categorical
from keras import initializers
from sklearn.metrics import classification_report, confusion_matrix
import pickle

import glob
import pandas as pd
import random
from keras.utils import multi_gpu_model
train = '../Input/train'
test = '../Input/test'

usemultiGPU = False
numGPU = 8
input_length = 22100*5
batch_size = 32 
eps =1e-8
Debugging=False
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

def GetCQT(audio, sample_rate=22100):
  
    cqt_spec = librosa.core.cqt(y=audio,sr=sample_rate,fmin=librosa.note_to_hz('C2'),n_bins=60 * 2,bins_per_octave=12 * 2)
    cqt_db = (librosa.amplitude_to_db(abs(cqt_spec), ref=np.max))
    return cqt_spec.T

def loadData(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=22100)[0] #, sr=16000
    
    if len(data)>input_length:    
        max_offset = len(data)-input_length        
        offset = np.random.randint(max_offset)        
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    data = GetCQT(data)
    #normalising 
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + eps
    data = np.divide(data - mean,std)
    #display_spectogram(data)
    return data

def GetConvModel():

    nclass = len(list_labels)
    inp = Input(shape=(216, 120, 1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64,)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    if usemultiGPU:
        parallel_model = multi_gpu_model(model, gpus=numGPU)
        parallel_model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        parallel_model.summary()
        return parallel_model

    else:
        model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model



train_files = glob.glob("../Input/train/*.wav")
test_files = glob.glob("../Input/test/*.wav")
train_labels = pd.read_csv("../Input/train.csv")
test_labels = pd.read_csv("../Input/test.csv")

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

model = GetConvModel()
model.load_weights("baseline_cnn_CQT.h5")
pred = []
groundtruth = []
index = 1
for name in val_files:
    print(index)
    data = [loadData(name)]
    data = np.array(data)[:, :, :,np.newaxis]
    Y_pred = model.predict(data)
    groundtruth.append(label_to_int[labels[name.split('\\')[-1]]])
    pred.append(np.argmax(Y_pred, axis=1))
    index=index+1
    
predarray = np.asarray(pred)
groundtrutharray = np.asarray(groundtruth)

    

print('Confusion Matrix')
print(confusion_matrix(groundtrutharray, predarray))
print('Classification Report')
print(classification_report(groundtrutharray, predarray))