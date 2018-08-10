"""Trains and Evaluates the SMART network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import argparse

from smarter import input_data as in_data
from smarter import smarter_common as common

logs_dir = 'logs/'

  

def run_bulk_inference(model, data_sets, k=1):

  if model=='os':
    from smarter import smarter_model_os as model
  elif model=='service':
    from smarter import smarter_model_service as model
  elif model=='cause':
    from smarter import smarter_model_cause as model
  else:
    print('Model not supported', model)
    exit()  
  
  #num_features = model.get_num_features()

  bulk_size = data_sets.test.labels.shape[0]
  print('Bulk size:', bulk_size)
  
  """Infer from previouslly saved SMART model using the test data set"""
  # Tell TensorFlow that the model will be built into the default Graph.
  #with tf.Graph().as_default():
  # Generate placeholders for the features and labels.
  features_placeholder, labels_placeholder = common.placeholder_inputs(bulk_size)

  # Build a Graph that computes predictions from the inference model.
  logits = model.inference(features_placeholder)

  if k==0:
    # Add the Op that retuns the predicted labels
    labels_predict = model.make_predictions(logits)
  else:
    #Add the Op to get multi top statistics
    eval_correct = model.evaluation_stats(logits, labels_placeholder, k)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()

  # Create a saver for reading training checkpoints.
  saver = tf.train.Saver(sharded=True)

  # Create a session for running Ops on the Graph.
  sess = tf.Session()

  # Run the Op to initialize the variables.
  tf.global_variables_initializer().run(session=sess)

  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

  ckpt = tf.train.get_checkpoint_state(logs_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print('No checkpoint found...')

  # Now you can run the model to get predictions
  print('Test Data Eval:')
  if k==0:
    common.do_stats(sess, labels_predict, features_placeholder, labels_placeholder, data_sets.test)
  else:
    common.do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.test)


def run_inference(model, features):

  import logging
  
  # Get an instance of a logger
  logger = logging.getLogger(__name__)
  
  k = 5; #for top_k results

  if model=='os':
    from smarter import smarter_model_os as model
  elif model=='service':
    from smarter import smarter_model_service as model
  elif model=='cause':
    from smarter import smarter_model_cause as model
  else:
    print('Model not supported', model)
    exit()  

  # Generate placeholders for the features and labels.
  features_placeholder = tf.placeholder(tf.float32, [1, None]) #FLAGS.features]) #, labels_placeholder = placeholder_inputs(1)

  # Build a Graph that computes predictions from the inference model.
  logits = model.inference(features_placeholder)
  
  #Transform logits into probabilities
  softmax = tf.nn.softmax(logits)
  
  # Get the correct position
  #correct = tf.argmax(logits, 1)
  
  # Get the top-k correct positions
  top_k_prob, top_k_idx = tf.nn.top_k(softmax, k)
  
  # Create a saver for reading training checkpoints.
  saver = tf.train.Saver(sharded=True)

  # Create a session for running Ops on the Graph.
  sess = tf.Session()

  # Run the Op to initialize the variables.
  tf.global_variables_initializer().run(session=sess)

  ckpt = tf.train.get_checkpoint_state(logs_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("Sessions are here: ", logs_dir)
    print('No checkpoint found...')

  # Now you can run the model to get predictions
  #top_k, index, softmax = sess.run([corrrect_top_k, correct, softmax], feed_dict={features_placeholder: features})    
  prob, index, softmax = sess.run([top_k_prob, top_k_idx, softmax], feed_dict={features_placeholder: features})    
  #prob = softmax[0][index]
  
  #print('Top', k,'answers are', index[0], 'with confidence', prob[0])
  
  return index[0], prob[0]

