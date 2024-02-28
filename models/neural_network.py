import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split


def model_builder(neurons, learning_rate_, lambda_,input_shape_):   
    
    model = keras.Sequential()
    
    
    model.add(InputLayer(input_shape=(input_shape_)))
    
    for i in range(len(neurons)-1):
        model.add(keras.layers.Dense(units=neurons[i], activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(lambda_))
                  )
    
    model.add(keras.layers.Dense(units=neurons[-1],
                                 kernel_regularizer=keras.regularizers.l2(lambda_))
              )       

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_),
                metrics=['accuracy']
)
    return model