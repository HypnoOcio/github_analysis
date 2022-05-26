import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional,Dropout, Dense, Concatenate, LSTM, BatchNormalization 
from tensorflow.keras.layers import MaxPooling1D, Flatten, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import Sequence,to_categorical
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

import numpy as np
import random

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, batch_size=4, dim=None, n_classes=2, shuffle=True, months_cnt = None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = data.sort_values(by = ['project_id','created_at'])
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.months_cnt = months_cnt
        self.on_epoch_end()
        self.number_of_months_to_end = 24
        self.num_of_timeseries = len( [col for col in data.columns if "_count" in col] )
        self.name_of_timeseries = [col for col in data.columns if "_count" in col]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) 

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        x_data = []
        # Generate data
        trans = StandardScaler()
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            df_repo = self.df[self.df.project_id == ID]
            for col in self.name_of_timeseries:
              x = ( np.array( df_repo[col].iloc[:self.months_cnt] ) )
              # Noramlize data 0 - 1 range
              # x_data.append( (x - np.min(x)) / (np.max(x) - np.min(x)) if np.min(x) != np.max(x) else x )
              
              # Standardization
              stand_x = trans.fit_transform(x.reshape(-1, 1))
              new_data = trans.fit_transform(stand_x)
              x_data.append( new_data )

            # Store class
            y[i] = 0 if len(df_repo) - self.months_cnt < self.number_of_months_to_end else 1 
        # Add padding
        X = tf.keras.preprocessing.sequence.pad_sequences(x_data, maxlen = self.months_cnt, padding="pre", dtype='float',)
        X = X.reshape(self.batch_size, self.months_cnt, self.num_of_timeseries )
        return X, to_categorical(y, num_classes=self.n_classes)