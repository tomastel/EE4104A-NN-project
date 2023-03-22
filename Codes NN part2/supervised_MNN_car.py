''' MNN example: car fuel efficiency (MPG)'''

import keras
from keras.models import Sequential
from keras.layers import Dense

#import data from the file
from numpy import genfromtxt
data = genfromtxt('auto-mpg.data', skip_header=0, usecols=range(0,8)) # import columns 0 to 7 (8 columns in total)

# create the array of expected output from the "data" array. 1st ":" means all rows; 2nd ":" means from column 0 to column 1-1=0
y_expect = data[:,0:1]

# create the array of input. 2nd ":" means from column 0 to column 8-1=7
x_input = data[:,1:8]

# rescale the input
#for i in range(0,8):
#    x_input[:,i:i+1] = x_input[:,i:i+1]/max(x_input[:,i:i+1])


# specify the network structure. Dense() means fully connected layer
n1 = 7 # size of input layer
n2 = 2 # size of hidden layer
n3 = 1 # size of output layer
model = Sequential() # create the network layer by layer
keras.initializers.RandomUniform(minval=-0.1, maxval=0.1) # initialize the weights
model.add(Dense(units=n2, input_dim=n1, activation='sigmoid')) # hidden layer, logistic function
model.add(Dense(units=n3, activation='linear')) # output layer, linear function


# choose loss function, optimization method, and metrics to evaluate
# loss function: mean squared error
# optimization method: gradient descent
# metrics to evaluate: mean squared error
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=['mean_squared_error'])

# train the model; batch_size = how many samples in each mini batch
train_history = model.fit(x=x_input, y=y_expect, validation_split=0, epochs=300, batch_size=20, verbose=2)

# plot the metrics to show the training results
import matplotlib.pyplot as plt
def show_train_history(train_history, train):
    plt.plot(train_history.history[train])  
    #plt.title('Train History', fontsize=20)  
    plt.ylabel(train, fontsize=20)  
    plt.xlabel('Epoch', fontsize=20)  
    plt.show()
show_train_history(train_history, 'mean_squared_error')
