# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:28:19 2019

@author: vmarimut
"""

import numpy as np # linear algebra
import librosa
import pandas as pd
import glob
import scipy


filename = "../Input/train.csv"

trainpath = '../Input/'
label = 'Drawer_open_or_close'
EPS = 1e-8
train_files = glob.glob(trainpath+label+"/*.wav")


#should be run after FilterForDataAugmentation.py. it creates augmented data for file

#test 


Speedy_files=[]
pitchy_files=[]

for file_path in train_files:
    #time shift
    wav, sr = librosa.load(file_path, sr=None)
    speed_change = np.random.uniform(low=0.9,high=1.5)
    maxv = np.iinfo(np.int16).max
    y = librosa.effects.time_stretch(wav, speed_change)
    savepath = trainpath+label+'/time_shift_'+file_path.split('\\')[-1]
    scipy.io.wavfile.write(savepath, sr, (y*maxv).astype(np.int16))
    Speedy_files.append(['time_shift_'+file_path.split('\\')[-1],label])
    
    #pitch shift
    y_pitch = wav.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    print("pitch_change = ",pitch_change)
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), 
                                      sr, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
   
    savepath = trainpath+label+'/pitch_shift'+file_path.split('\\')[-1]
    scipy.io.wavfile.write(savepath, sr, (y*maxv).astype(np.int16))
    pitchy_files.append(['pitch_shift'+file_path.split('\\')[-1],label])


for case_row in Speedy_files:
    df = pd.DataFrame(case_row).T
    try:
        with open(filename, 'a') as f:
            df.to_csv(f, header=False, index=False)
    except:
        print('Unable to Write CSV')
        
for case_row in pitchy_files:
    df = pd.DataFrame(case_row).T
    try:
        with open(filename, 'a') as f:
            df.to_csv(f, header=False, index=False)
    except:
        print('Unable to Write CSV')
