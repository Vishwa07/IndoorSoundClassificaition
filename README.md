# IndoorSoundClassificaition


Spectrograms extracted from sound have been useful in CNN based architectures to classify sound. 
Spectrograms such as short-time Fourier transform(STFT) and constant Q transform(CQT) represent a good representation of the temporal and spectral structure of the original audio.
In this paper, STFT and CQT features extracted from an audio file were assessed for the classification of various sound in the dataset using CNNs.
The experiment shows that 89% train accuracy,87% validation accuracy, and 78% test accuracy were achieved for the FSDKaggle2018 dataset


STFT&CQT.py corresponds to training the main model of the project . 
Results of the model are under results folder of the project in output.csv
DataAuguementation does time shift and pitch shift for the specified lables and updates the train.csv
FilterForDataAugmetation.py filters the specified audio and moves to destination folder
TestSTFt.py outputs the predicted class code along with confusion matrix and classification report.
Misc folder contains all the experimented code 
