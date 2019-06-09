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
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPool2D, GlobalMaxPool2D,Activation,Flatten,concatenate
from sklearn.model_selection import train_test_split
from keras.utils import Sequence, to_categorical
from keras import initializers
import glob
import pandas as pd
import random
from keras.utils import multi_gpu_model
train = '../Input/train'
test = '../Input/test'

usemultiGPU = False
numGPU = 8
input_length = 22100*5
batch_size = 128 
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
            
def display_spectogram(log_S):
    sr = 22100
    plt.figure(figsize=(14,5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.title('cqt_note spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    
def GetCQT(audio, sample_rate=22100):
    
   
    cqt_spec = librosa.core.cqt(y=audio,sr=sample_rate,fmin=librosa.note_to_hz('C2'),n_bins=60 * 2,bins_per_octave=12 * 2)
    cqt_db = (librosa.amplitude_to_db(abs(cqt_spec), ref=np.max))
    #display_spectogram(cqt_db)
    #print(cqt_db.shape)
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
    
    #data = data.reshape(data.shape[0],1).T    
    data = GetCQT(data)
    #print(data.shape)
   # data = GetGammatone(data)
    #display_spectogram(data)
    #print(int_to_label[file_to_int[os.path.basename(file_path)]])
    #display_spectogram(data)
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
    opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999)
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

tr_files, val_files = train_test_split(sorted(train_files), test_size=0.2, random_state=42)



#loadData(os.path.join(train, '01f2e70b.wav'))    
#loadData(os.path.join(train, '00c934d7.wav'))    
#loadData(os.path.join(train, '00d1fe46.wav'))    
#loadData(os.path.join(train, '00d40fa2.wav'))   
#PREDICTION_FOLDER = "predictions_2d_conv"
#if not os.path.exists(PREDICTION_FOLDER):
#    os.mkdir(PREDICTION_FOLDER)
#if os.path.exists('logs/' + PREDICTION_FOLDER):
#    shutil.rmtree('logs/' + PREDICTION_FOLDER)
#
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model = GetConvModel()
model.load_weights("baseline_cnn_CQT.h5")
layer_names = []

if Debugging:    
    for layer in model.layers[2:11]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    print(layer_names)
    layer_outputs = [layer.output for layer in model.layers[2:11]]
    inputdata = model.input
    activation_model = models.Model(inputs=inputdata, outputs=layer_outputs)
    testdata = loadData(os.path.join(train, '00ad7068.wav'))
    testbatch_data = [testdata]
    testbatch_data = np.array(testbatch_data)[:, :, :,np.newaxis]
    activations_debugging = activation_model.predict(testbatch_data)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations_debugging): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        size1 = layer_activation.shape[2]
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size1))
        print(layer_activation.shape)
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size1 : (row + 1) * size1] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1]*2,
                            scale * display_grid.shape[0]*2))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()



history = model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=20,
                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,
                   use_multiprocessing=True, workers=8, max_queue_size=20,
                    callbacks=[ModelCheckpoint("baseline_cnn_CQT.h5", monitor="val_acc", save_best_only=True)],verbose=1)
                               #EarlyStopping(patience=5, monitor="val_acc")],verbose=1)
 
model.save_weights("baseline_cnn_CQT.h5")

fig =  plt.figure(figsize=(12,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ioff()
plt.savefig("accuracy.png") 
plt.close(fig)


# summarize history for loss
fig =  plt.figure(figsize=(12,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ioff()
plt.savefig("loss.png") 
plt.close(fig)
# 
