# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:25:14 2019

@author: vmarimut
"""

from gammatone import gtgram
from gammatone import plot
import matplotlib.pyplot as plt
import numpy as np
import librosa

class GammatoneFilterbank:
    sample_rate = 16000
    window_time = 0.025
    hop_time = 0.010
    num_filters = 320
    cutoff_low = 30
    def __init__(self, sample_rate=16000, window_time=0.025, hop_time= 0.010, num_filters=320, cutoff_low=30):
        self.sample_rate = sample_rate
        self.window_time = window_time
        self.hop_time = hop_time
        self.num_filters = num_filters
        self.cutoff_low = cutoff_low

    def make_spectrogram(self, audio_samples):
        gtg=gtgram.gtgram(audio_samples,
                             self.sample_rate,
                             self.window_time,
                             self.hop_time,
                             self.num_filters,
                             self.cutoff_low)
      
        return gtg
        

        
    
  
        
    def make_dilated_spectral_frames(self, audio_samples, num_frames, dilation_factor):
        spectrogram = self.make_spectrogram(audio_samples)
        spectrogram = np.swapaxes(spectrogram, 0, 1)
        dilated_frames = np.zeros((len(spectrogram), num_frames, len(spectrogram[0])))

        for i in range(len(spectrogram)):
            for j in range(num_frames):
                dilation = np.power(dilation_factor, j)

                if i - dilation < 0:
                    dilated_frames[i][j] = spectrogram[0]
                else:
                    dilated_frames[i][j] = spectrogram[i - dilation]

        return dilated_frames