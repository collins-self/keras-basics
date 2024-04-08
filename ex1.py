# Program that finds the mathematical relationship between x and y
import numpy as np
from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Sequential is how you define layers of a Neural Network,
#   Layer Type: Dense - layer is one where each neuron is connected to each in the
#       next layer (units is how many neurons in the Dense Layer, 1)
#   Input Shape: Shape of the first layer (we only have one) aka what the input looks
#       like, just a single value (1,)
model = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])
# The computer needs to guess the rel between x and y. Then needs to see how good
#   or bad that guess is (loss function). The optimizer makes the next guess.
# Stochastic Gradient Descent takes in the previous values and loss, generates
# another guess.
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

slope = model.layers[0].get_weights()[0]
weight = model.layers[0].get_weights()[1]

# The machine learns the correlation between the two numbers
print(model.predict(np.array([10.0])))

print("Based on dataset: \nX = {}\nY = {}".format(xs, ys))
print("Model learned Y = {}X + {}".format(slope, weight))
