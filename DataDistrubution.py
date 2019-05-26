# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:27:16 2019

@author: vmarimut
"""

import librosa
import librosa.display
import glob
import scipy.io.wavfile
from scipy.fftpack import dct
import pandas as pd
labels = dict()
last_name = dict()
import matplotlib.pylab as plt

train_files = glob.glob("../Input/train/*.wav")
train_labels = pd.read_csv("../Input/train.csv")

sample_each_label = dict()
for name,label  in zip(train_labels.fname.values,train_labels.label.values):
    labels[name]=label
    sample_each_label[label] = name


for name in train_files:
    fname = name.split('\\')[-1]
    last_name[name] = fname
    sample_rate, signal = scipy.io.wavfile.read(name,True)
    print(len(signal))
    #print(labels[fname] + " "+str(sample_rate))

category_count = dict()
for name in train_files:
    label = labels[last_name[name]]
    if label in category_count:
        category_count[label] = category_count[label] + 1
    else:
        category_count[ label] = 1
#sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory

print(category_count)
print(sample_each_label)