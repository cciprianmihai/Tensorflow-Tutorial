import tensorflow as tf

matrix_1 = tf.constant([[2, 2]])
matrix_2 = tf.constant([[3],[3]])

dot_operation = tf.matmul(matrix_1, matrix_2)

# The result is not computed yet
print(dot_operation)

# Create tensorflow session
sess = tf.Session()

# Call dot_operation within the session
result = sess.run(dot_operation)

# Print the result
print(result)

# Close the session
sess.close()

# Create tensorflow session
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)