"""Builds the SMART network model that predicts SMART_SR_REGISTRY.DIAGNOSIS_CODE
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
This file is used by "smarter.py" file and not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

learning_rate = 0.01

def get_num_classes():
  return 182

def get_num_hidden1():
  return 400

def get_num_hidden2():
  return 200

def get_num_features():
  return 562 #14 + 338 + 210
  #return 385 #14 + 161 + 210

def weight_variable(shape):
  initial = tf.random_normal(shape)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.random_normal(shape)
  return tf.Variable(initial)


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


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations


def inference(features):
  """Build the SMART model up to where it may be used for inference.
  Args:
    features: input placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Store layers weight & bias

  #weights = {
    #'h1': tf.Variable(tf.random_normal([get_num_features(), get_num_hidden1()])),
    #'h2': tf.Variable(tf.random_normal([get_num_hidden1(), get_num_hidden2()])),
    #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    #'out': tf.Variable(tf.random_normal([get_num_hidden2(), get_num_classes()]))
  #}
  #biases = {
    #'b1': tf.Variable(tf.random_normal([get_num_hidden1()])),
    #'b2': tf.Variable(tf.random_normal([get_num_hidden2()])),
    #'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    #'out': tf.Variable(tf.random_normal([get_num_classes()]))
  #}

  #layer_1 = tf.nn.tanh(tf.add(tf.matmul(features, weights['h1']), biases['b1'])) #Hidden layer with TANH activation
  #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) #Hidden layer with RELU activation
  #layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])) #Hidden layer with RELU activation


  layer_1 = nn_layer(features, get_num_features(), get_num_hidden1(), 'layer1', act=tf.nn.tanh)
  #layer_2 = nn_layer(layer_1, get_num_hidden1(), get_num_hidden2(), 'layer2', act=tf.nn.tanh)

  #with tf.name_scope('dropout'):
  #  keep_prob = tf.placeholder(tf.float32)
  #  keep_prob = 0.1
  #  tf.scalar_summary('dropout_keep_probability', keep_prob)
  #  dropped = tf.nn.dropout(layer_1, keep_prob)

  # Do not apply softmax activation yet, see below.
  logits = nn_layer(layer_1, get_num_hidden1(), get_num_classes(), 'out', act=tf.identity)

  #logits = tf.matmul(layer_1, weights['out']) + biases['out']
  return logits


def loss(logits, labels):
  """
  Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.classes].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)

  tf.summary.scalar(   'cross_entropy', cross_entropy)

  return cross_entropy


def training(loss):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  with tf.name_scope('train'):
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss) #, global_step=global_step)

  return train_op


def eval_bad_data(logits, labelled_sessions, k=10):
  """
  Identifies sessions that's very hard to infer the correct cause.
  Note that k value should be as high as possible so that only a few sessions are identified
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.classes].
    labelled_sessions: Labels and sessions tensor, int32 - [batch_size, 2], with values for index 0 in the range [0, FLAGS.classes) and having index 1 hold the session id.
    k: number of top predictions
  Returns:
    A scalar int32 tensor with the bad session ids.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a scalar int32 tensor with shape [batch_size] that holds
  # ids for bad sessions
  correct = tf.nn.in_top_k(logits, labelled_sessions[0], k)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def evaluation_stats(logits, labels, k=1):
  """
  Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.classes].
    labels: Labels tensor, int32 - [batch_size], with values in the range [0, FLAGS.classes).
    k: number of top predictions
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct = tf.nn.in_top_k(logits, labels, k)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
      #accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
  tf.summary.scalar('accuracy', accuracy*100)
  # Return the number of true entries.
  return accuracy*labels.get_shape()[0].value #tf.reduce_sum(tf.cast(correct, tf.int32))


def make_predictions(logits):
  """
  Make predictions for the all the data.
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.classes].
  Returns:
    A int32 tensor with the predictions (of batch_size).
  """

  #Transform logits into probabilities
  softmax = tf.nn.softmax(logits)

  # Get the predicted positions, index
  predicted = tf.argmax(logits, 1)

  return predicted