# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:18:17 2019

@author: vmarimut
"""


import shutil, os
import glob
import pandas as pd

train = '../Input/train'
train_files = glob.glob("../Input/train/*.wav")
test_files = glob.glob("../Input/test/*.wav")
train_labels = pd.read_csv("../Input/train.csv")
labels = dict()
fileList= []
labelToFilter = 'Meow'
#for name,label  in zip(train_labCoels.fname.values,train_labels.label.values):
#    labels[name]=label


#move all files of a particular label to destination directory
for name,label  in zip(train_labels.fname.values,train_labels.label.values):
    if label == labelToFilter:
        fileList.append(name)
    labels[name]=label

directory = '../Input/' + labelToFilter
if not os.path.exists(directory):
    os.mkdir(directory)
for f in fileList:
    f= '../Input/train/'+f
    shutil.copy(f, directory)
