import tensorflow as tf

# Define variables in the "global_variable" set
var_1 = tf.Variable(1)
var_2 = tf.Variable(2)
# Define the operation between the variable and
add_operation = tf.add(var_1, var_2)
update_operation = tf.assign(var_1, add_operation)

with tf.Session() as sess:
    # Initialize the variables using the global_variables_initializer() method
    sess.run(tf.global_variables_initializer())
    # Call the update operation - var_1 will be the result of add_operation(var_1, var_2)
    sess.run(update_operation)
    # Print var_1 value
    print(sess.run(var_1))