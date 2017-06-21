#!/usr/bin/env python

import tensorflow as tf
import numpy as np

sess = tf.Session()

# Create some random variables
x_input = np.random.random_sample(200)
y_input = x_input * 20 + 6

# Parameter layer of model
with tf.name_scope('Parameter_Layer'):
    W = tf.Variable([.5], name='Weights')
    b = tf.Variable([.6], name='Bias')

# Input layer of model
with tf.name_scope('Input_Layer'):
    x_data = tf.placeholder(tf.float16, name='X_Input')
    linear_model = tf.multiply(W, x_data, name='W*x') + b

# Output layer of model
with tf.name_scope('Output_Layer'):
    y_data = tf.placeholder(tf.float16, name='Y_Output')

# Evaluation layer of model
with tf.name_scope('Evaluation_Layer'):
    squared_deltas = tf.square(linear_model - y_data, name='Squared_deltas')
    loss = tf.reduce_sum(squared_deltas)

# Train layer of mode
with tf.name_scope('Evaluation_Layer'):
    optimizer = tf.train.GradientDescentOptimizer(0.2, name='Learning_speed_0.2')
    train = optimizer.minimize(loss, name='Minimize_loss')

merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorflow/linear_regression', sess.graph)
init = tf.global_variables_initializer()

sess.run(init)
for i in range(500):
    sess.run(train, x_input, y_input)

print(sess.run([W, b]))
