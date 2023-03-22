''' MNN example: digit recognition, with input rescaled and training/test split'''

import keras
from keras.models import Sequential
from keras.layers import Dense

# import data from the file (columns 0 to 793=(28*28=10-1))
# columns 0 to 784-1 are input (28*28=784 columns)
# columns 784 to 793 are output (10 columns)
# delimiter=',' is to specify that the columns are separated by ","
from numpy import genfromtxt
data = genfromtxt('mnist_1k.csv', delimiter=',', skip_header=0, usecols=range(0,794))

x_input = data[:,0:784] # columns 0 to 784-1 are input
x_input = x_input/255 # rescale the input to [0,1]
y_expect = data[:,784:794] # columns 784 to 793 are output


# specify the network structure. Dense() means fully connected layer
n1 = 28*28 # size of input layer
n2 = 15 # size of hidden layer
n3 = 10 # size of output layer
model = Sequential() # create the network layer by layer
keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
model.add(Dense(units=n2, input_dim=n1, activation='sigmoid')) # hidden layer
model.add(Dense(units=n3, activation='sigmoid')) # output layer

# choose loss function, optimization method, and metrics to evaluate
# loss function: squared difference
# optimization method: gradient descent
# metrics to evaluate: mean squared error and 'categorical_accuracy'
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.1), metrics=['mean_squared_error', 'categorical_accuracy'])

# "validation_split" specifics the split between training and validation/test
# validation_split=0.2 means 80% of data is used for training and 20% for test
train_history = model.fit(x=x_input, y=y_expect, validation_split=0.2, epochs=5000, batch_size=25, verbose=2)

# plot the metrics to show the training results
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History', fontsize=20)  
    plt.ylabel(train, fontsize=20)
    plt.xlabel('Epoch', fontsize=20)  
    plt.legend(['Train', 'Validation / Test'], loc='best', fontsize=16) 
    plt.show()

# plot both the training and validation/test results; "val_xxx" means "validation"
show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error')
show_train_history(train_history, 'categorical_accuracy', 'val_categorical_accuracy')

