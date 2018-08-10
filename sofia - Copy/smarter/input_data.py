"""Functions for reading SMART data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy
from numpy import genfromtxt
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



PATH = './'
#featuresSize = 385
featuresSize = 562
labelsSize = 1
#numLines = 30388
numLines = 76871
batch_size = 500
train_dir='data/'

# set defaults to something (TF requires defaults for the number of cells you are going to read)
rDefaults = [tf.constant([0],dtype=tf.int64) for row in range(labelsSize)] + [tf.constant([0],dtype=tf.float32) for row in range(featuresSize)]



def extract_batches(filename):
  #filename_queue = tf.train.string_input_producer([train_dir +"train-data-000.csv",train_dir +"train-data-001.csv",train_dir +"train-data-002.csv",train_dir +"train-data-003.csv",
#train_dir +"train-data-004.csv",train_dir +"train-data-005.csv",train_dir +"train-data-006.csv",train_dir +"train-data-007.csv",train_dir +"train-data-008.csv",train_dir +"train-data-009.csv",
#train_dir +"train-data-010.csv",train_dir +"train-data-011.csv",train_dir +"train-data-012.csv",train_dir +"train-data-013.csv",train_dir +"train-data-014.csv",train_dir +"train-data-015.csv",
#train_dir +"train-data-016.csv",train_dir +"train-data-017.csv",train_dir +"train-data-018.csv",train_dir +"train-data-019.csv"])

  print('Extracting', filename)
  reader = tf.TextLineReader()
  filename_queue = tf.train.string_input_producer([filename])
  _, csv_row = reader.read(filename_queue) # read one line
  data = tf.decode_csv(csv_row, record_defaults=rDefaults) # use defaults for this line (in case of missing data)

  labels = data[0]
  features = data[labelsSize:labelsSize+featuresSize]

  # minimum number elements in the queue after a dequeue, used to ensure
  # that the samples are sufficiently mixed
  # I think 10 times the BATCH_SIZE is sufficient
  min_after_dequeue = 10 * batch_size

  # the maximum number of elements in the queue
  capacity = 20 * batch_size

  # shuffle the data to generate BATCH_SIZE sample pairs
  features_batch, labels_batch = tf.train.shuffle_batch([features, labels], batch_size=batch_size, num_threads=10, capacity=capacity, min_after_dequeue=min_after_dequeue)
  #features_batch, labels_batch = tf.train.batch([features, labels], batch_size=batch_size)


  return features_batch, labels_batch


def extract_data(filename, one_hot=False):
  """Extract the images into a 2D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  raw_data = genfromtxt(filename, dtype=numpy.int32, delimiter=',')
  features = raw_data[:, 1:]
  labels = raw_data[:, 0]

  if one_hot:
    labels = dense_to_one_hot(labels)

  return features, labels


def dense_to_one_hot(labels_dense, num_classes=267):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):
  def __init__(self, features, labels):
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = features.shape[0]

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
	end = self._num_examples
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]




def read_data_sets(train_dir, one_hot=False):
  class DataSets(object):
    pass

  data_sets = DataSets()

  TRAIN_DATA_FILE = 'train-data.csv'
  TEST_DATA_FILE = 'test-data.csv'
  VALIDATION_SIZE = 300

  train_features, train_labels = extract_data(train_dir + TRAIN_DATA_FILE, one_hot=one_hot)  # first column is label
  test_features, test_labels = extract_data(train_dir + TEST_DATA_FILE, one_hot=one_hot)  # first column is label

  validation_features = train_features[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  # train_features = train_features[:VALIDATION_SIZE]
  # train_labels = train_labels[:VALIDATION_SIZE]

  data_sets.train = DataSet(train_features, train_labels)
  data_sets.validation = DataSet(validation_features, validation_labels)
  data_sets.test = DataSet(test_features, test_labels)

  print('Train data set shape:', data_sets.train.features.shape)
  print('Validation data set shape:', data_sets.validation.features.shape)
  print('Test data set shape:', data_sets.test.features.shape)

  return data_sets
