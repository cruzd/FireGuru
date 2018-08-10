"""Builds the SMART network model.
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


learning_rate=0.001

def get_num_classes():
  return 30

def get_num_hidden1():
  return 80

def get_num_features():
  return 186

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
  weights = {
    'h1': tf.Variable(tf.random_normal([get_num_features(), get_num_hidden1()])),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([get_num_hidden1(), get_num_classes()]))
  }
  biases = {
    'b1': tf.Variable(tf.random_normal([get_num_hidden1()])),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([get_num_classes()]))
  }
  layer_1 = tf.nn.relu(tf.add(tf.matmul(features, weights['h1']), biases['b1'])) #Hidden layer with RELU activation
  #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) #Hidden layer with RELU activation
  #layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])) #Hidden layer with RELU activation	
  
  logits = tf.matmul(layer_1, weights['out']) + biases['out']
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
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

  
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
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """
  Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.classes].
    labels: Labels tensor, int32 - [batch_size], with values in the range [0, FLAGS.classes).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
