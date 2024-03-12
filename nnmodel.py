"""
Name: nnmodel
Artifical neural network

Inputs:
    X-train, y_train, X_test

Output:
    trained model

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import initializers
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from keras import backend as K

#%% Self defined coefficient of determination function
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%% We need to standardize the dataset (IMP)
def standardize(X_train, X_val, X_test):
    sc = MinMaxScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_val_std = sc.transform(X_val)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_val_std, X_test_std

#%% Neural network
def neuralnet(X, y, X_test):
    
    from sklearn.model_selection import train_test_split
    
    # Lets split the dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

    # Standardize the data
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    ## Neural Network Model
    
    # reduced LR on plateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.0000000001)
    
    # initialization of the 0th interation weights
    initializer =  tf.keras.initializers.GlorotNormal(seed=201)
    
    X_rows, input_shape = np.shape(X_train)
    inputs = keras.Input(shape=(input_shape,)) # 
    x = layers.Dense(32, activation='tanh', kernel_initializer=initializer)(inputs)  
    x = Dropout(0.2)(x)
    x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(x)
    x = Dropout(0.2)(x)
    x = layers.Dense(32, activation='tanh', kernel_initializer=initializer)(x)
    #x = layers.Dense(64, activation='sigmoid', kernel_initializer=initializer)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    
    # Model:
    model = keras.Model(inputs,outputs, name='model')
    
    # Optimzer: Here used (Adam): https://keras.io/api/optimizers/
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    
    # Compile model with loss function as mean squared error
    # metrics determines the training root mean square error as well as R2 values.
    # The last epoch values are the final RMSE and R2.
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(), coeff_determination]
                  )

    # Fit model for train data and validate it on validation data
    model.fit(X_train,y_train,
              validation_data=(X_val,y_val),
              batch_size=32,epochs=10, callbacks=[reduce_lr])
    model.summary()
    
    return model, X_test, X_train