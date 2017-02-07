""" CIFAR-10 Dataset

Credits: A. Krizhevsky. https://www.cs.toronto.edu/~kriz/cifar.html.

"""
from __future__ import absolute_import, print_function

import os
import sys
from six.moves import urllib
import tarfile
import math
import numpy as np
import pickle
import tensorflow as tf
import numpy

def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def load_data(dirname="cifar-10-batches-py", one_hot=False):
    tarpath = maybe_download("cifar-10-python.tar.gz",
                             "http://www.cs.toronto.edu/~kriz/",
                             dirname)
    X_train = []
    Y_train = []

    if dirname != './cifar-10-batches-py':
        dirname = os.path.join(dirname, 'cifar-10-batches-py')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    # 2D + 2D + 2D = 3D

    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if one_hot:
        Y_train = to_categorical(Y_train, 10)
        Y_test = to_categorical(Y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading CIFAR 10, Please wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
        print(('Succesfully downloaded', filename, statinfo.st_size, 'bytes.'))
        untar(filepath)
    return filepath

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def untar(fname):
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall(path = '/'.join(fname.split('/')[:-1]))
        tar.close()
        print("File Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2] * images.shape[3])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      # images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_images, train_labels, test_images, test_labels):
  class DataSets(object):
    pass
  data_sets = DataSets()
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def build_weight(shape):
    return tf.Variable(tf.random_normal(shape))


def weight_FC(shape):
    weight = tf.Variable(tf.truncated_normal(shape=shape))
    return weight

def bias_FC(shape):
    bias = tf.Variable(tf.zeros(shape=shape))
    return bias

def inference(image):
    image = tf.reshape(image, [-1, 32, 32, 3])

    # Conv1
    W1 = build_weight([3, 3, 3, 32])
    b1 = build_weight([32])
    conv1 = conv2d(image, W1, b1)

    # max pool
    max_pool1 = maxpool2d(conv1, 2)

    # Conv2
    W2 = build_weight([3, 3, 32, 64])
    b2 = build_weight([64])
    conv2 = conv2d(max_pool1, W2, b2)

    # Conv3
    W3 = build_weight([3, 3, 64, 64])
    b3 = build_weight([64])
    conv3 = conv2d(conv2, W3, b3)

    # max pool 2
    max_pool2 = maxpool2d(conv3, 2)

    # FC
    dim = 1
    for d in max_pool2.get_shape()[1:].as_list():
        dim *= d

    max_pool2_reshape = tf.reshape(max_pool2, [96, dim])
    w_fc_1 = weight_FC([dim, 512])
    b_fc_1 = bias_FC([512])

    fc_1 = tf.matmul(max_pool2_reshape, w_fc_1) + b_fc_1
    fc_1 = tf.nn.relu(fc_1)
    fc_1_dropout = tf.nn.dropout(fc_1, 0.5)  # dropout

    # FC 2
    w_fc_2 = weight_FC([512, 10])
    b_fc_2 = bias_FC([10])
    fc_2 = tf.matmul(fc_1_dropout, w_fc_2) + b_fc_2

    # logits = tf.nn.softmax(fc_2)
    logits = fc_2
    # with tf.name_scope('hidden1'):
    #     weight = tf.Variable(tf.truncated_normal([dim, 512], name='weight'))
    #     bias = tf.Variable(tf.zeros([512]), name='bias')
    #     hidden1 = tf.nn.relu(tf.matmul(max_pool2_reshape, weight) + bias)
    #     hidden1 = tf.nn.dropout(hidden1, 0.5)
    #
    # with tf.name_scope('hidden2'):
    #     weight = tf.Variable(tf.truncated_normal([512, 10], name='weight'))
    #     bias = tf.Variable(tf.zeros([10]), name='bias')
    #     hidden2 = tf.matmul(hidden1, weight) + bias
    #
    # with tf.name_scope('softmax'):
    #     logits = tf.nn.softmax(hidden2)

    return logits


def loss(logits, label):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    return cost


def train_op(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def accuracy(logits, label):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(label, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc



