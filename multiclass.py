import pandas as pd
from collections import Counter

from keras import utils

import multiprocessing
import os
import librosa
import numpy as np
import keras
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from keras.layers.recurrent import LSTM
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
from sklearn.utils import class_weight


def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=22050))
    

def normalize_mfcc(mfcc):
    
    
    v_min = mfcc.min(axis=(0, 1), keepdims=True)
    v_max = mfcc.max(axis=(0, 1), keepdims=True)
    mfcc = (mfcc - v_min)/(v_max - v_min)
    
#    def normalize(x, axis=0):
#    return sklearn.preprocessing.minmax_scale(x, axis=axis)

    
    
    return(mfcc)
    
def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    COL_SIZE = 30
    
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)
    
def train_model_MLP(X_train,y_train):
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    
    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    
    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(200, input_dim=rows*cols, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(lr=0.1),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,validation_split = 0.2, epochs=300, batch_size=64, class_weight=class_weights)
    
    return (model)

def train_model_CNN(X_train,y_train): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]


    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0003),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,validation_split = 0.1, epochs=100, batch_size=32, class_weight=class_weights)

    return (model)

def train_model_RNN(X_train,y_train): #64
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(np.array(X_train).shape[1],np.array(X_train).shape[2])))
    model.add(LSTM(64, return_sequences=True))
    
    model.add(LSTM(64))
    model.add(Dense(6, activation='relu'))
    # add dropout to control for overfitting
    model.add(Dropout(.25))
    
    # squash output onto number of classes in probability space
    model.add(Dense(6, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0003),
                  metrics=['accuracy'])
    
        
    history = model.fit(np.asarray(X_train).astype(float), np.asarray(y_train).astype(float), validation_split = 0.1,epochs=150, batch_size=128, class_weight=class_weights )

    return (model)


if __name__ == '__main__':

#########################################################################

    df = pd.read_csv('C:/Users/Ahmed/Desktop/dissertation/archive/recordings/recordings_wave/wave.csv')
    
    
    #### labels at the moment are just english vs not english that are converted to 0 for not english and 1 for english
    labels = pd.read_csv("C:/Users/Ahmed/Desktop/dissertation/archive/labels_5.csv") 
    labels.loc[(labels.native_language=='not_english'), 'native_language'] = 0
    labels.loc[(labels.native_language=='english'), 'native_language'] = 1
    labels.loc[(labels.native_language=='spanish'), 'native_language'] = 2
    labels.loc[(labels.native_language=='arabic'), 'native_language'] = 3
    labels.loc[(labels.native_language=='mandarin'), 'native_language'] = 4
    labels.loc[(labels.native_language=='french'), 'native_language'] = 5
    labels = np.array(labels)


    mfccs = []
    for i in range(len(df)):
        mfcc = to_mfcc(np.array(df.iloc[i]))  
        mfccs.append(mfcc)
#            
#    mfccs1 = normalize_mfcc(np.array(mfccs))
##    
#    mfccs2, labels1 = make_segments(mfccs1,labels)
    
    X_train, X_test, y_train, y_test = train_test_split(mfccs,labels , test_size=0.3, random_state=42, shuffle = True, stratify =labels)


#    class_weights = class_weight.compute_class_weight('balanced',
#                                                     np.unique(np.ravel(y_train,order='C')),
#                                                     np.ravel(y_train,order='C'))
    
    
    class_weights = {0: 0.30546624,
                    1: 0.61555076,
                    2: 2.19230769,
                    3: 3.47560976,
                    4: 5.48076923,
                    5: 5.7 }

 
    model  = train_model_MLP(np.array(X_train).astype(float), np.array(y_train).astype(float))
    model  = train_model_CNN(np.array(X_train).astype(float), np.array(y_train).astype(float))
    model  = train_model_RNN(np.array(X_train).astype(float), np.array(y_train).astype(float))
    
    
    ## this plots the accuracy and val_accuarcy
    plt.plot(model.history.history['accuracy'])  
    plt.plot(model.history.history['val_accuracy'])
    plt.title('LSTM model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
    rows = np.array(X_test[0]).shape[0]
    cols = np.array(X_test[0]).shape[1]
    X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], rows, cols, 1 )
    y_test = y_test.astype(float)
    predictions =  np.argmax(model.predict(np.array(X_test).astype(float)), axis=1)
    
    rows = np.array(X_test[0]).shape[0]
    cols = np.array(X_test[0]).shape[1]
    X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], rows, cols )
    y_test = y_test.astype(float)
    predictions =  np.where(np.argmax(model.predict(np.array(X_test).astype(float)), axis=1)>1,1,0)
    
    
    
    print('test accuracy = ',accuracy_score(y_test, predictions))
    print('test precision = ',precision_score(y_test, predictions, average='weighted'))
    print('test recall = ',recall_score(y_test, predictions, average='weighted'))
    print('test f1 = ',f1_score(y_test, predictions, average='macro'))
    print('matthsew correlation = ',matthews_corrcoef(y_test, predictions))
    
    
    
    cf = confusion_matrix(y_test, predictions)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);  
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Other', 'English','Spanish', 'Arabic','Mandarin','French']); ax.yaxis.set_ticklabels(['Other', 'English','Spanish', 'Arabic','Mandarin','French'], rotation=45);



## Error anlaysis 
#errors = (predictions- np.array(y_test).astype(float).reshape(428,) !=0).astype(float)
#total_errors = np.sum(errors==True)
#print(total_errors)
#errors_index = np.array(np.where(errors==True)).reshape(116,)
