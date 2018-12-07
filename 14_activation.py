import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the data
x = np.linspace(-5, 5, 200)

# Plot the most popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

# Softmax is a special kind of activation function, it is about probability
# y_softmax = tf.nn.softmax(x)

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

# Plot to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
# 2 2 1 - 2 2 2 - 2 2 3 - 2 2 4 - We have 4 figures (we make 4 spaces for them, and we put them in order 1, 2, 3, 4)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()