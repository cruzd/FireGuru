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
#from tensorflow_serving.session_bundle import exporter

import input_data
import smart_model
import argparse


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', -1, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.  Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data/', 'Directory to read the training data.')
flags.DEFINE_string('logs_dir', 'logs/', 'Directory to read the training data.')
flags.DEFINE_integer('export_version', 1, 'Version for exporting')
flags.DEFINE_string('export_dir', 'export/', 'Directory where to export the model for serving')

def placeholder_inputs(batch_size):
  """
  Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the csv data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    features_placeholder: Features placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the features
  # and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  features_placeholder = tf.placeholder(tf.float32, [batch_size, smart_model.FEATURES_SIZE])
  labels_placeholder = tf.placeholder(tf.int64, [batch_size])
  return features_placeholder, labels_placeholder


def fill_feed_dict(data_set, features_pl, labels_pl):
  """
  Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of features and labels, from input_data.read_data_sets()
    features_pl: The features placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next `batch size` examples.
  features_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  
  #print data to see it's ok
  #print('features shape: ', features_feed.shape)
  #print('labels shape: ', labels_feed.shape)
  
  #for i in xrange(2):
  #  print(features_feed[i], labels_feed[i])  
    
  feed_dict = {
      features_pl: features_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

"""
def export4serving(sess, saver, input_tensor, scores_tensor):
  export_path = FLAGS.export_dir
  print('Exporting trained model to', export_path)
  model_exporter = exporter.Exporter(saver)
  signature = exporter.classification_signature(input_tensor=input_tensor, scores_tensor=scores_tensor)
  model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
  model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)  
"""  
  
  
def do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_set):
  """
  Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    features_placeholder: The features placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of features and labels to evaluate, from input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, features_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision))


def run_training():
  """Train SMART for a number of steps."""
  # Get the sets of features and labels for training, validation, and test on SMART data.
  data_sets = input_data.read_data_sets(FLAGS.train_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the features and labels.
    features_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = smart_model.inference(features_placeholder, FLAGS.hidden1) #, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = smart_model.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = smart_model.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = smart_model.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(sharded=True)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of features and labels for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train, features_placeholder, labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.logs_dir, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.test)
        
        #export4serving(sess, saver, features_placeholder, labels_placeholder)


def run_inference():

  """Infer from previouslly saved SMART model."""
  # Get the sets of features and labels for training, validation, and test on SMART data.
  data_sets = input_data.read_data_sets(FLAGS.train_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the features and labels.
    features_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = smart_model.inference(features_placeholder, FLAGS.hidden1) #, FLAGS.hidden2)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = smart_model.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for reading training checkpoints.
    saver = tf.train.Saver(sharded=True)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, graph_def=sess.graph_def)

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint found...')

    # Now you can run the model to get predictions
    print('Test Data Eval:')
    do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.test)



def main(_):
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument('-m', help='mode train or infer', choices=['train','infer', 'overtrain'])
  args = parser.parse_args()  
  
  if args.m=='train':  
    run_training()
  if args.m=='infer':
    run_inference()
  
  
if __name__ == '__main__':
  tf.app.run()