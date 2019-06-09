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
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import random
from gammatone_filterbank import GammatoneFilterbank
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report, confusion_matrix
train = '../Input/train'
test = '../Input/test'

usemultiGPU = False
numGPU = 8
input_length = 16000*5
batch_size = 32 
filterbank = GammatoneFilterbank()
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
            
def display_spectogram(log_S):
    sr = 16000
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='linear')
    plt.title('gamma spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    
def GetGammatone(audio, sample_rate=16000, window_size=20, #log_specgram
                 step_size=10, eps=1e-10):
    
   
    gamma = filterbank.make_spectrogram(audio)
    spec = (librosa.amplitude_to_db(gamma, ref=np.max))
    #display_spectogram(spec)

    #filterbank.GetGammotoneSpectogram(audio)
    #log_specgram = np.log(audio.T.astype(np.float32) + eps)
    return spec.T

def loadData(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
    
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
    
    #data = data.reshape(data.shape[0],1).T    
    data = GetGammatone(data)
    #print(data.shape)
   # data = GetGammatone(data)
    #display_spectogram(data)
    #print(int_to_label[file_to_int[os.path.basename(file_path)]])

    return data

def GetConvModel():
    nclass = len(list_labels)
    inp = Input(shape=(498, 320, 1))
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


#
#loadData(os.path.join(train, '00ad7068.wav'))    
#loadData(os.path.join(train, '00c934d7.wav'))    
#loadData(os.path.join(train, '00d1fe46.wav'))    
#loadData(os.path.join(train, '00d40fa2.wav'))   


model = GetConvModel()
model.load_weights("baseline_cnn_gammatone.h5")

model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=10,
                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,
                   use_multiprocessing=False, workers=8, max_queue_size=60,
                    callbacks=[ModelCheckpoint("baseline_cnn_gammatone.h5", monitor="val_acc", save_best_only=True),
                               EarlyStopping(patience=5, monitor="val_acc")])
model.save_weights("baseline_cnn_comb.h5")
    
#pred = []
#groundtruth = []
#index=1
#for name in val_files:
#    print(index)
#    data = [loadData(name)]
#    data = np.array(data)[:, :, :,np.newaxis]
#    Y_pred = model.predict(data)
#    groundtruth.append(label_to_int[labels[name.split('\\')[-1]]])
#    pred.append(np.argmax(Y_pred, axis=1))
#    index=index+1
#    
#
#predarray = np.asarray(pred)
#groundtrutharray = np.asarray(groundtruth)
#
#    
#
#print('Confusion Matrix')
#print(confusion_matrix(groundtrutharray, predarray))
#print('Classification Report')
#print(classification_report(groundtrutharray, predarray))
# 
# 
