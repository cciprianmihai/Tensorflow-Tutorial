import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Generating data
x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1) - column vector
noise = np.random.normal(0, 0.1, size=x.shape) # shape (100, 1) - column vector
y = np.power(x,2) + noise # shape (100, 1) - column vector

def save():
    print("This is save step")
    # Define the Neural Network layers
    tf_x = tf.placeholder(tf.float32, x.shape) # Placeholder for computational graph with shape (100, 1)
    tf_y = tf.placeholder(tf.float32, y.shape) # Placeholder for computational graph with shape (100, 1)
    layer_1 = tf.layers.dense(tf_x, 10, tf.nn.relu) # Hidden layer
    layer_2 = tf.layers.dense(layer_1, 1)           # Output layer
    loss = tf.losses.mean_squared_error(tf_y, layer_2) # Compute the cost
    training = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    # Create the session - control training
    sess = tf.Session()
    # Initialize all the graph variables
    sess.run(tf.global_variables_initializer())
    # Define a saver for saving and restoring
    saver = tf.train.Saver()
    # Training the network and output the result
    for step in range(100):
        sess.run(training, {tf_x: x, tf_y: y})

    saver.save(sess, './23_save_reload', write_meta_graph=False)

    # Plotting the results
    pred, l = sess.run([layer_2, loss], {tf_x: x, tf_y: y})
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Save Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})


def reload():
    print("This is the reload step")
    # Define the Neural Network layers
    tf_x = tf.placeholder(tf.float32, x.shape) # Placeholder for computational graph with shape (100, 1)
    tf_y = tf.placeholder(tf.float32, y.shape) # Placeholder for computational graph with shape (100, 1)
    layer_1 = tf.layers.dense(tf_x, 10, tf.nn.relu) # Hidden layer
    layer_2 = tf.layers.dense(layer_1, 1) # Output layer
    loss = tf.losses.mean_squared_error(tf_y, layer_2) # Compute the cost
    # Create the session - control training
    sess = tf.Session()
    # Don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, './23_save_reload')

    # Plotting the results
    pred, l = sess.run([layer_2, loss], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()

save()
# destroy previous net
tf.reset_default_graph()
reload()
