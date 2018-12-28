import tensorflow as tf
import numpy as np


# Load your data or create your data in here
# x data
npx = np.random.uniform(-1, 1, (1000, 1))
# y data
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)
# Training and test data
npx_train, npx_test = np.split(npx, [800])
npy_train, npy_test = np.split(npy, [800])

# Use placeholder, later you may need different data, pass the different data into placeholder
tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

# Create dataloader
dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
# Choose data randomly from this buffer
dataset = dataset.shuffle(buffer_size=1000)
# Batch size you will use
dataset = dataset.batch(32)
# Repeat for 3 epochs
dataset = dataset.repeat(3)
# Later we have to initialize this one
iterator = dataset.make_initializable_iterator()

# Define the network
# Use batch to update
bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# Create the session - control training
sess = tf.Session()
# Need to initialize the iterator in this case
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})

for step in range(201):
    try:
        # train
        _, trainl = sess.run([train, loss])
        if step % 10 == 0:
            # test
            testl = sess.run(loss, {bx: npx_test, by: npy_test})
            print('step: %i/200' % step, '|train loss:', trainl, '|test loss:', testl)
    # If training takes more than 3 epochs, training will be stopped
    except tf.errors.OutOfRangeError:
        print('Finish the last epoch.')
        break