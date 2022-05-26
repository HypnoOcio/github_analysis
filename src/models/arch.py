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

def cnn_architecture(num_classes, n_timesteps, n_features):

  cnn_input = Input(shape=(n_timesteps,n_features),name="CNN_INPUT")
  con_1D_one = Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features), name = "CONVOLUTION_LAYER_1") (cnn_input)
  con_1D_one = BatchNormalization()(con_1D_one)
  con_1D_one = Dropout(0.2)(con_1D_one)
  max_pool_one = MaxPooling1D(pool_size=3, name = "MAXPOOL_LAYER_1")(con_1D_one)

  con_1D_two = Conv1D(filters=64, kernel_size=3, activation='relu', name = "CONVOLUTION_LAYER_2") (max_pool_one)
  con_1D_two = BatchNormalization()(con_1D_two)
  con_1D_two = Dropout(0.2)(con_1D_two)
  max_pool_two = MaxPooling1D(pool_size=2, name = "MAXPOOL_LAYER_2")(con_1D_two)

  flatten_layer_three = Flatten(name = "FLATTEN_LAYER_3")(max_pool_two)
  dense_layer_four = Dense(100, activation='relu', name = "DENSE_LAYER_4")(flatten_layer_three)
  output = Dense(num_classes,activation=softmax,name="OUTPUT_LAYER_5")(dense_layer_four)

  model = Model(inputs=[cnn_input],outputs=[output])# binary cross entropy loss
  return model

def rnn_architecture(num_classes, n_timesteps, n_features):
  recurrent_input = Input(shape=(n_timesteps,n_features),name="TIMESERIES_INPUT")
  rec_layer_one = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),return_sequences=True),name ="BIDIRECTIONAL_LAYER_1")(recurrent_input)
  rec_layer_one = BatchNormalization()(rec_layer_one)
  rec_layer_one = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),return_sequences=True),name ="BIDIRECTIONAL_LAYER_X")(rec_layer_one)
  rec_layer_one = BatchNormalization()(rec_layer_one)
  rec_layer_two = Bidirectional(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),name ="BIDIRECTIONAL_LAYER_2")(rec_layer_one)
  rec_layer_two = BatchNormalization()(rec_layer_two)
  rec_layer_two = Dense(64, activation='relu', name = "DENSE_LAYER_3")(rec_layer_two)
  output = Dense(num_classes,activation=softmax,name="OUTPUT_LAYER")(rec_layer_two)
  model = Model(inputs=[recurrent_input],outputs=[output])
  return model