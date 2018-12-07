import tensorflow as tf

# Define the placeholders
x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
# Define the operation between the placeholders
z1 = x1 + y1

# Define the placeholders
x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
# Define the operation between the placeholders
z2 = tf.matmul(x2, y2)

# Start the session
with tf.Session() as sess:
    # Only one operation to run - feed the dictionary with values
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})
    print(z1_value)

    # Multiple operations to run - feed the dictionary with values
    z1_value, z2_value = sess.run(
        [z1, z2],
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z1_value)
    print(z2_value)