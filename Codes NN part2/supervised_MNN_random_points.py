import keras
from keras.models import Sequential
from keras.layers import Dense

#import data from the file (columns 0 to 4-1=3); skipping the header
from numpy import genfromtxt
data = genfromtxt('random_points.txt', skip_header=1, usecols=range(0,4))

x_input = data[:,0:2] # in the "data" array, columns 0 to 2-1=1 is the input
y_expect = data[:,2:4] # columns 2 to 4-1=3 is the output

# specify the network structure. Dense() means fully connected layer
n1 = 2 # size of input layer
n2 = 10 # size of hidden layer
n3 = 2 # size of output layer
model = Sequential() # create the network layer by layer
keras.initializers.RandomUniform(minval=-0.1, maxval=0.1) # initialize the weights
model.add(Dense(units=n2, input_dim=n1, activation='sigmoid')) # hidden layer, logistic function
model.add(Dense(units=n3, activation='sigmoid')) # output layer, logistic function

# choose loss function, optimization method, and metrics to evaluate
# loss function: squared difference
# optimization method: gradient descent
# metrics to evaluate: mean squared error and 'categorical_accuracy'
# 'categorical_accuracy' will automatically convert the output from probabilities to the guessed class, and count the % of correct guesses
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=5), metrics=['mean_squared_error', 'categorical_accuracy'])

train_history = model.fit(x=x_input, y=y_expect, validation_split=0, epochs=300, batch_size=20, verbose=2)

# plot the metrics to show the training results
import matplotlib.pyplot as plt
def show_train_history(train_history, train):
    plt.plot(train_history.history[train])
    plt.title('Train History', fontsize=20)  
    plt.ylabel(train, fontsize=20)  
    plt.xlabel('Epoch', fontsize=20)  
    plt.show()
show_train_history(train_history, 'mean_squared_error')
plt.figure()
show_train_history(train_history, 'categorical_accuracy')

