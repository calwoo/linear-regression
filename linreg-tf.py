# A version of linear regression using tensorflow.

import numpy as np 
import tensorflow as tf 
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

## Now we set up our data
num_data = 30
noise = 0.3
X_data, y_data = generate_data(3, 2, noise, num_data)

# Hyperparameters
learning_rate = 0.5
num_episodes = 100

# We create a basic computational graph for our linear regression.
X = tf.placeholder("float")
W = tf.get_variable(name="weight", initializer=tf.constant(np.random.randn()))
b = tf.get_variable(name="bias", initializer=tf.constant(np.random.randn()))
y_guess = X * W + b
y = tf.placeholder("float")

# Loss
loss = tf.reduce_mean(tf.pow(y_guess - y, 2))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

## Run model
sess = tf.Session()
sess.run(init)
for i in range(num_episodes):
    sess.run(optimizer, feed_dict={X: X_data, y: y_data})
    if i % 10 == 0:
        episode_loss = sess.run(loss, feed_dict={X: X_data, y: y_data})
        print("Episode ", i, " ==> Loss: ", episode_loss,
            "  w: ", sess.run(W), "  b: ", sess.run(b))

final_w = sess.run(W)
final_b = sess.run(b)
plt.scatter(X_data, y_data)
x_plot = np.linspace(0,1,30)
plt.plot(x_plot, 3*x_plot+2, '-g', label="real line")
plt.plot(x_plot, final_w*x_plot+final_b, '-r', label="predicted line")
plt.legend()
plt.show()
