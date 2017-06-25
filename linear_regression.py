#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# Create some random variables
x_input = 10 * np.random.random_sample([2000])
y_input = x_input * 20 + 6

# Parameter layer of model
with tf.name_scope('Parameter_Layer'):
    W = tf.Variable([16], dtype=tf.float32, name='Weights')
    variable_summaries(W)
    b = tf.Variable([6], dtype=tf.float32, name='Bias')
    variable_summaries(b)

# Input layer of model
with tf.name_scope('Input_Layer'):
    x_data = tf.placeholder(tf.float32, 2000, name='X_Input')
    linear_model = W * x_data + b

# Output layer of model
with tf.name_scope('Output_Layer'):
    y_data = tf.placeholder(tf.float32, 2000, name='Y_Output')

# Evaluation layer of model
with tf.name_scope('Evaluation_Layer'):
    squared_deltas = tf.square((linear_model - y_data), name='Squared_deltas')
    loss = tf.reduce_sum(squared_deltas)
    tf.summary.scalar('Loss', loss)

# Train layer of mode
with tf.name_scope('Train_Layer'):
    optimizer = tf.train.GradientDescentOptimizer(0.000001, name='Learning_speed')
    train = optimizer.minimize(loss, name='Minimize_loss')

init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/tensorflow/linear_regression/train', sess.graph)

sess.run(init)
for i in range(4000):
    sess.run(train, {x_data: x_input, y_data: y_input})
    summary, acc = sess.run([merged, loss], {x_data: x_input, y_data: y_input})
    train_writer.add_summary(summary, i)

print(sess.run([W, b]))
print(sess.run(loss, {x_data:x_input, y_data:y_input}))

train_writer.close()
