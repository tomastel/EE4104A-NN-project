# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:18:36 2021

@author: tomas
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plt
import time

start_time = time.time()

data = np.genfromtxt('C:/Users/tomas/Desktop/Skole/Elektroingeniør/5. semester/EE4014A Intelligent Systems Applications in Electrical Engineering/Project/wine_data.txt', delimiter=',')

y_expected = data[:,0]
x_input = data[:,1:15]

rescaling_x = True
if rescaling_x:
    for i in range(0,13):
        x_input[:,i] = x_input[:,i]/max(x_input[:,i]) 

for j in range(0, len(y_expected)):
    y_expected[j] -= 1

y_expected_onehot = keras.utils.to_categorical(y_expected)

model = Sequential()
keras.initializers.RandomUniform(minval=-1/math.sqrt(13), maxval=-1/math.sqrt(13))
model.add(Dense(units=8, input_dim=13, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=['mean_squared_error'])

training_history = model.fit(x=x_input, y=y_expected_onehot, validation_split=0.2, epochs=300, batch_size=30, verbose=2)

tot_time = str(round(time.time()-start_time, 4))
print('\n--- ' + tot_time + ' seconds ---')

def plot_training_hist(training_history, training, validation, time):
    plt.plot(training_history.history[training])
    plt.plot(training_history.history[validation])
    plt.title('Training History', fontsize=20)
    plt.ylabel('Mean squared error', fontsize=12)  
    plt.xlabel('Epoch', fontsize=12)  
    plt.legend(['Training', 'Test'], loc='best')
    plt.text(250, 0.3, ('Runtime:\n'+time+' s'))
    plt.show()
plot_training_hist(training_history, 'mean_squared_error', 'val_mean_squared_error', tot_time)
