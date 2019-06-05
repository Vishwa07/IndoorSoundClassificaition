# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:07:58 2019

@author: vmarimut
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from multiprocessing.pool import ThreadPool

input_length = 16000*5
filters = 320

def display_spectogram(log_S,name,label):
 try:
    
    fig = plt.figure(figsize=(0.1 * log_S.shape[1]*2,
                            0.1 * log_S.shape[0]*2))
    sr = 16000
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.title('cqt_note spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    imagename = name+"_"+label+".png"
    imagename = "../Input/train/" + imagename
    
    print(imagename)
    #plt.show()
    plt.ioff()
    plt.savefig(imagename) 
    plt.close(fig)
 except:
     print("An exception occurred for " + name)
   
    
def GetCQT(audio,filepath,sample_rate=16000, window_size=20, #log_specgram
                 step_size=10, eps=1e-10):
    
   
        
        cqt_spec = librosa.core.cqt(y=audio,sr=16000, n_bins=60)
        cqt_db = (librosa.amplitude_to_db(cqt_spec, ref=np.max))
        name = filepath.split('\\')[-1]
        mean = np.mean(cqt_db, axis=0)
        std = np.std(cqt_db, axis=0) + eps
        cqt_db = np.divide(cqt_db - mean,std)
        display_spectogram(cqt_db.T,name.replace(".wav",""),labels[name])
       
        return cqt_db.T
  
    #display_spectogram(cqt_db)

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
    data = GetCQT(data,file_path)

train_labels = pd.read_csv("../Input/train.csv")
train_files = glob.glob("../Input/train/*.wav")

labels= dict()
for name,label  in zip(train_labels.fname.values,train_labels.label.values):
    labels[name]=label


train_files = glob.glob("../Input/train/*.wav")
#results = ThreadPool(8).imap_unordered(loadData, train_files)

for name in train_files:
    data = loadData(name)
