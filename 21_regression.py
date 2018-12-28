import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Generating data
x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1) - column vector
noise = np.random.normal(0, 0.1, size=x.shape) # shape (100, 1) - column vector
y = np.power(x,2) + noise # shape (100, 1) - column vector

# Plot the data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape) # Placeholder for computational graph with shape (100, 1)
tf_y = tf.placeholder(tf.float32, y.shape) # Placeholder for computational graph with shape (100, 1)

# Define the Neural Network layers
layer_1 = tf.layers.dense(tf_x, 10, tf.nn.relu) # Hidden layer
layer_2 = tf.layers.dense(layer_1, 1) # Output layer

# Compute the cost
loss = tf.losses.mean_squared_error(tf_y, layer_2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
training = optimizer.minimize(loss)

# Create the session - control training
sess = tf.Session()
# Initialize all the graph variables
sess.run(tf.global_variables_initializer())

plt.ion()
for step in range(100):
    # Training the network and output the result
    _, l, pred = sess.run([training, loss, layer_2], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        # Plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()