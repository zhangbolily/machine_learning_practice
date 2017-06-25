from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import os

PROJECT_DIR = "/home/ballchang/PycharmProjects/machine_learning_practice/MNIST_data"

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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# First, I want to create a data set to evaluate the input data.
# This data set based on MNIST data set.The structure of the data set is
# a three dimension array.
# This function return an array of data set.
# Structure:[Index, num, data]
#          :"Index" is the identifier of data.
#          :"num" is the type of number.
#          :"data" is the data of number image.

def construct_data_set():
    scale = 10000

    _data_set_ = np.zeros((scale, 784))
    _label_set_ = np.zeros((scale, 10))
    _data_temp_ = np.zeros(784, dtype=int)

    if os.path.isfile(PROJECT_DIR + "/mnist_data_in_bit.data") == 1:
        if os.path.isfile(PROJECT_DIR + "/mnist_label_in_bit.data") == 1:
            batch_data = np.loadtxt(PROJECT_DIR + "/mnist_data_in_bit.data")
            batch_label = np.loadtxt(PROJECT_DIR + "/mnist_label_in_bit.data")
            _data_set_ = np.reshape(batch_data, (scale, 784))
            _label_set_ = np.reshape(batch_label, (scale, 10))
            print("[Function Message]In construct_data_set(): Read samples from files for testing.\n")
            return _data_set_, _label_set_

    batch_data, batch_label = mnist.train.next_batch(55000)

    k = 0
    for i in range(10):
        for j in range(55000):
            if batch_label[j, i] == 1 and k < (scale/10*(i + 1)):
                for l in range(784):
                    _data_temp_[l] = 1 if batch_data[j, l] > 0.8 else 0

                # print(_data_temp_.reshape((28, 28)))
                _data_set_[k] = _data_temp_
                _label_set_[k] = batch_label[j]
                # print(_label_set_[k], "\n")
                k += 1
    print("[Function Message]In construct_data_set(): Constructed ", k, " samples for testing.\n")

    np.savetxt(PROJECT_DIR + "/mnist_data_in_bit.data", _data_set_)
    np.savetxt(PROJECT_DIR + "/mnist_label_in_bit.data", _label_set_)

    return _data_set_, _label_set_


def classify_type(var1, var2, var3):
    loss_set = np.zeros(10000)
    result = np.zeros(10)

    # print(var3[0], "\n")

    difference = np.tile(var1, (np.shape(var2)[0], 1)) - var2
    # print(difference[1], "\n")

    difference_square = difference ** 2
    # print(difference_square[1], "\n\n")

    loss_set = difference_square.sum(axis=1) ** 0.5
    # print(loss_set, "\n\n")
    result_index = np.argsort(loss_set)

    for i in range(20):
        result = result + var3[result_index[i]]
        # print(var3[result_index[i]])

    type_index = np.argsort(result)

    # print(result, loss_set[result_index[19]], type_index[9])

    return type_index[9]


def construct_test_set():
    _test_data_set_ = np.zeros((400, 784))
    _test__label_set_ = np.zeros((400, 10))
    _data_temp_ = np.zeros(784, dtype=int)

    if os.path.isfile(PROJECT_DIR + "/mnist_testdata_in_bit.data") == 1:
        if os.path.isfile(PROJECT_DIR + "/mnist_testlabel_in_bit.data") == 1:
            batch_data = np.loadtxt(PROJECT_DIR + "/mnist_testdata_in_bit.data")
            batch_label = np.loadtxt(PROJECT_DIR + "/mnist_testlabel_in_bit.data")
            _test_data_set_ = np.reshape(batch_data, (400, 784))
            _test__label_set_ = np.reshape(batch_label, (400, 10))
            print("[Function Message]In construct_test_set(): Read samples from files for testing.\n")
            return _test_data_set_, _test__label_set_

    _test_data_set_, _test__label_set_ = mnist.test.next_batch(400)
    for i in range(400):
        for j in range(784):
            _data_temp_[j] = 1 if _test_data_set_[i, j] > 0.8 else 0
        # print(_data_temp_.reshape((28, 28)))
        _test_data_set_[i] = _data_temp_

    np.savetxt(PROJECT_DIR + "/mnist_testdata_in_bit.data", _test_data_set_)
    np.savetxt(PROJECT_DIR + "/mnist_testlabel_in_bit.data", _test__label_set_)

    return _test_data_set_, _test__label_set_


data_set, label_set = construct_data_set()
test_data, test_label = construct_test_set()

accuracy = 0

for i in range(400):
    result = classify_type(test_data[i], data_set, label_set)
    if np.argmax(test_label[i]) == result:
        # print(np.argmax(test_label[i]), classify_type(test_data[i], data_set, label_set))
        print("The original number is ", np.argmax(test_label[i]), ". The result is ", result, ".\n")
        accuracy += 1

print("The accuracy rate is ", accuracy/400, ".\n")
