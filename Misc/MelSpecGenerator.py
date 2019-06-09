# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:50:10 2019

@author: vmarimut
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import random

input_length = 16000*5
n_mels = 320

def display_spectogram(log_S,name,label):
    sr = 22050
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('log mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.savefig(name+"_"+label) 
    
    
def GetLogMelSpec(audio, sample_rate=16000, window_size=20, #log_specgram
                 step_size=10, eps=1e-10):

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40    #log_specgram = np.log(mel_db.T.astype(np.float32) + eps)


    return mel_db.T

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
    data = GetLogMelSpec(data)

train_labels = pd.read_csv("../Input/train.csv")
train_files = glob.glob("../Input/train/*.wav")

labels= dict()
for name,label  in zip(train_labels.fname.values,train_labels.label.values):
    labels[name]=label


train_files = glob.glob("../Input/train/*.wav")
for name in train_files:
    data = loadData(name)
    display_spectogram(data,name,labels[name.split('\\')[-1]])
