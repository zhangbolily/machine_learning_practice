from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import os

PROJECT_DIR = "/home/ballchang/PycharmProjects/machine_learning_practice/MNIST_data"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

gray = 0.2
test_count = 20000
k = 3

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

# First, I want to create a data set to evaluate the input data.
# This data set based on MNIST data set.The structure of the data set is
# a three dimension array.
# This function return an array of data set.
# Structure:[Index, num, data]
#          :"Index" is the identifier of data.
#          :"num" is the type of number.
#          :"data" is the data of number image.


def construct_data_set():
    scale = 20000

    _data_set_ = np.zeros((scale, 784), dtype=int)
    _label_set_ = np.zeros((scale, 10), dtype=int)
    _data_temp_ = np.zeros(784, dtype=int)

    # If data files already exist, just load them.
    if os.path.isfile(PROJECT_DIR + "/mnist_data_binary.npy") == 1:
        if os.path.isfile(PROJECT_DIR + "/mnist_label_binary.npy") == 1:
            batch_data = np.load(PROJECT_DIR + "/mnist_data_binary.npy")
            batch_label = np.load(PROJECT_DIR + "/mnist_label_binary.npy")
            _data_set_ = np.reshape(batch_data, (scale, 784))
            _label_set_ = np.reshape(batch_label, (scale, 10))
            print("[Function Message]In construct_data_set(): Read samples from files for testing.\n")
            return _data_set_, _label_set_

    # Read data from MNIST
    batch_data, batch_label = mnist.train.next_batch(55000)

    # Binarizing data set
    k = 0
    for i in range(10):
        for j in range(55000):
            if batch_label[j, i] == 1 and k < (scale/10*(i + 1)):
                for l in range(784):
                    _data_temp_[l] = 1 if batch_data[j, l] > gray else 0

                # print(_data_temp_.reshape((28, 28)))
                _data_set_[k] = _data_temp_
                _label_set_[k] = batch_label[j]
                # print(_label_set_[k], "\n")
                k += 1
    print("[Function Message]In construct_data_set(): Constructed ", k, " samples for testing.\n")

    # Save data array for next time using
    np.save(PROJECT_DIR + "/mnist_data_binary.npy", _data_set_)
    np.save(PROJECT_DIR + "/mnist_label_binary.npy", _label_set_)

    return _data_set_, _label_set_


def construct_test_set():
    scale = 20000
    _test_data_set_ = np.zeros((scale, 784), dtype=int)
    _test__label_set_ = np.zeros((scale, 10), dtype=int)
    _data_temp_ = np.zeros(784, dtype=int)

    # If data files already exist, just load them.
    if os.path.isfile(PROJECT_DIR + "/mnist_testdata_binary.npy") == 1:
        if os.path.isfile(PROJECT_DIR + "/mnist_testlabel_binary.npy") == 1:
            batch_data = np.load(PROJECT_DIR + "/mnist_testdata_binary.npy")
            batch_label = np.load(PROJECT_DIR + "/mnist_testlabel_binary.npy")
            _test_data_set_ = np.reshape(batch_data, (scale, 784))
            _test__label_set_ = np.reshape(batch_label, (scale, 10))
            print("[Function Message]In construct_test_set(): Read samples from files for testing.\n")
            return _test_data_set_, _test__label_set_

    # Read data from MNIST
    _test_data_set_, _test__label_set_ = mnist.test.next_batch(scale)

    # Binarizing data set
    for i in range(scale):
        for j in range(784):
            _data_temp_[j] = 1 if _test_data_set_[i, j] > gray else 0
        # print(_data_temp_.reshape((28, 28)))
        _test_data_set_[i] = _data_temp_

    print("[Function Message]In construct_data_set(): Constructed ", scale, " samples for testing.\n")

    # Save data array for next time using
    np.save(PROJECT_DIR + "/mnist_testdata_binary.npy", _test_data_set_)
    np.save(PROJECT_DIR + "/mnist_testlabel_binary.npy", _test__label_set_)

    return _test_data_set_, _test__label_set_


def print_num(var1):
    temp = var1.reshape(784)
    print("Here is the image of this hand-write number.")
    for j in range(28):
        temp_print = ''
        for k in range(28):
            if temp[(j * k + k)] == 1:
                temp_print += '@'
                # temp_print[j] = '*'
            else:
                temp_print += ' '
                # temp_print[j] = ' '
        print(temp_print)


def print_result(var, index=0, var1=True):
    if var1 == 1:
        print("The original number is ", np.argmax(test_label[index]), ". The result is ", var, ".\n")
    else:
        print("The original number is ", np.argmax(test_label[index]), ". The result is ", var, ".")
        print("Index of the wrong answer is ", index, ".\n")


def classify_type(var1, var2, var3, var4):
    loss_set = np.zeros(20000)
    result = np.zeros(10)

    difference = np.tile(var1, (np.shape(var2)[0], 1)) - var2
    # print(difference[1], "\n")

    difference_square = difference ** 2
    # print(difference_square[1], "\n\n")

    loss_set = difference_square.sum(axis=1)
    # print(loss_set, "\n\n")
    result_index = np.argsort(loss_set)

    for i in range(var4):
        result = result + var3[result_index[i]]
        # print(loss_set[result_index[i]])

    type_index = np.argsort(result)

    # print(result, loss_set[result_index[19]], type_index[9])

    return type_index[9]


def classify_type_weights(var1, var2, var3, var4):
    loss_set = np.zeros(20000)
    result = np.zeros(10)

    difference = np.tile(var1, (np.shape(var2)[0], 1)) - var2
    # print(difference[1], "\n")

    difference_square = difference ** 2
    # print(difference_square[1], "\n\n")

    loss_set = difference_square.sum(axis=1)
    # print(loss_set, "\n\n")
    result_index = np.argsort(loss_set)

    for i in range(var4):
        result = result + var3[result_index[i]] / np.exp(loss_set[result_index[i]])
        # print(var3[result_index[i]])

    type_index = np.argsort(result)

    # print(result, loss_set[result_index[19]], type_index[9])
    return type_index[9]


data_set, label_set = construct_data_set()
test_data, test_label = construct_test_set()

np.set_printoptions(linewidth=120)

f = open(PROJECT_DIR + '/result.data', 'a+')

weight_accuracy = 0
accuracy = 0

# ------------------------------Calculate Code 1----------------------------------------

# f.write("Result of classify_type_weights function:\n")
#
# for j in range(test_count):
#     result = classify_type_weights(test_data[j], data_set, label_set, k)
#     if np.argmax(test_label[j]) == result:
#         weight_accuracy += 1
#
# print("The weight accuracy rate is ", weight_accuracy/test_count, ".K-value is ", k, ".\n")
# f.write("%f\n" % (weight_accuracy/test_count))

# ------------------------------Calculate Code 2----------------------------------------

f.write("Result of classify_type function:\n")

for j in range(test_count):
    result = classify_type(test_data[j], data_set, label_set, k)
    if np.argmax(test_label[j]) == result:
        accuracy += 1

print("The accuracy rate is ", accuracy/test_count, ".K-value is ", k, ".\n")
f.write("%f\n" % (accuracy / test_count))

# -------------------------------------End-----------------------------------------------

f.close()
