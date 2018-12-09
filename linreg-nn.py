# Now we want to try implementing linear regression using a neural network and gradient descent.

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# We emphasize that this is the stupidest possible neural network-- 1 input, 1 output, with 2 weights.
# I mean, this is linear regression after all...

# First, we need a function to generate a dataset to play with.
def generate_data(m, b, noise, num_data, seed=0):
    """
    INPUT
        m = slope of data
        b = bias of data
        noise = variance of Gaussian error terms
        num_data = number of data points
        seed = random seed
    OUTPUT
        X = list of input data values
        y = list of output data values
    """
    np.random.seed(seed)
    X =  np.random.uniform(0, 1, num_data)
    gaussian_noise = np.random.randn(num_data) * noise
    y = m * X + b + gaussian_noise
    return X, y

# We will train our neural network to minimize the L^2-loss.
def neural_net(X, w):
    """
    INPUT
        X = input data, np vector of pairs (1,x_i) for input.
        w = network weights (in this case, a pair of parameter values)
    OUTPUT
        y = output values
    """
    return X @ w

def loss(y_guess, y_real):
    """
    INPUT
        y_guess = output from neural network
        y_real = real outputs from dataset
    OUTPUT
        loss = average L^2-loss of dataset
    """
    L2_residuals = (y_guess - y_real)**2
    return L2_residuals.mean()

# Now we implement gradient descent updating.
def grad_loss(X, y, w):
    """
    INPUT
        X = input data, np vector of pairs (1, x_i) for input.
        y = output values from neural network
        w = network weights (in this case, a pair of parameter values)
    OUTPUT
        grad = gradient of loss with respect to weights
    """
    grad = -2*X.T @ (y - X@w)
    return grad

def update(X, y, w, learning_rate):
    """
    INPUT
        X, y, w = as expected
        learning_rate = learning rate
    OUTPUT
        new_w = new weights
        loss = loss of previous iteration
    """
    y_guess = neural_net(X, w)
    _loss = loss(y_guess, y)
    grad = grad_loss(X, y, w)
    new_w = w - learning_rate * grad
    return new_w, _loss
    
# Test grounds
n = 30
X_raw, y = generate_data(3, 2, 0.02, n)
X = np.c_[np.ones(n), X_raw]
w = np.random.randn(2)

learning_rate = 0.015
num_episodes = 100

for i in range(num_episodes):
    w, _loss = update(X, y, w, learning_rate)
    if i % 10 == 0:
        print("Episode ", i, ": loss = ", _loss, " weights = ", w)

# How did we do?
plt.scatter(X_raw, y)
plt.show()


