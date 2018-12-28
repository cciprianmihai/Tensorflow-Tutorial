import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Generating data
n_data = np.ones((100, 2))
x0 = np.random.normal(2 * n_data, 1) # class0 x with shape (100, 2)
y0 = np.zeros(100)                   # class0 y with shape (100, )
x1 = np.random.normal(-2 * n_data, 1)# class1 x with shape (100, 2)
y1 = np.ones(100)                    # class1 y with shape (100, )
x = np.vstack((x0, x1)) # shape (200, 2)
y = np.hstack((y0, y1)) # shape (200, )

# Plot the data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)# Placeholder for computational graph with shape (200, 2)
tf_y = tf.placeholder(tf.int32, y.shape)# Placeholder for computational graph with shape (200, )

# Define the Neural Network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)# Hidden layer
output = tf.layers.dense(l1, 2)# Output layer

# Compute the cost
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

# Create the session - control training
sess = tf.Session()
# Define variable for initialize all the graph variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# Initialize all the graph variables
sess.run(init_op)

plt.ion()
for step in range(100):
    # Training the network and output the result
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step % 2 == 0:
        # Plot and show learning process
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()