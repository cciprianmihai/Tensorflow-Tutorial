import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Generating data
x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1) - column vector
noise = np.random.normal(0, 0.1, size=x.shape) # shape (100, 1) - column vector
y = np.power(x,2) + noise # shape (100, 1) - column vector

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    layer_1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    layer_2 = tf.layers.dense(layer_1, 1, name='output_layer')
    # Add to histogram summary
    tf.summary.histogram('h_out', layer_1)
    tf.summary.histogram('pred', layer_2)

loss = tf.losses.mean_squared_error(tf_y, layer_2, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)     # add loss to scalar summary

# Create the session - control training
sess = tf.Session()
# Initialize all the graph variables
sess.run(tf.global_variables_initializer())
# Write to file
writer = tf.summary.FileWriter('./log', sess.graph)
# Operation to merge all summary
merge_op = tf.summary.merge_all()

for step in range(100):
    # Training the network and output the result
    _, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
    writer.add_summary(result, step)

# Lastly, in your terminal or CMD, type this: tensorboard --logdir <<path to log directory>>
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)