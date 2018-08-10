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
from tensorflow.contrib.tensorboard.plugins import projector
#from tensorflow_serving.session_bundle import exporter

from smarter import input_data as in_data
from smarter import smarter_common as common

max_steps = 15000
batch_size = 1000
num_features = -1
logs_dir = 'logs/'

"""
def export4serving(sess, saver, input_tensor, scores_tensor):
  export_path = FLAGS.export_dir
  print('Exporting trained model to', export_path)
  model_exporter = exporter.Exporter(saver)
  signature = exporter.classification_signature(input_tensor=input_tensor, scores_tensor=scores_tensor)
  model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
  model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)  
"""



def run_training(model, data_sets):

  if model=='os':
    from smarter import smarter_model_os as model
  elif model=='service':
    from smarter import smarter_model_service as model
  elif model=='cause':
    from smarter import smarter_model_cause as model
  else:
    print('Model not supported', model)
    exit()

  # Config to turn on JIT compilation
  tfconfig = tf.ConfigProto()
  tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  sess = tf.Session(config=tfconfig)

 # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

  features_placeholder, labels_placeholder = common.placeholder_inputs(batch_size)

  # Build a Graph that computes predictions from the inference model.
  logits = model.inference(features_placeholder)

  # Add to the Graph the Ops for loss calculation.
  loss_op = model.loss(logits, labels_placeholder)

  # Add to the Graph the Ops that calculate and apply gradients.
  train_op = model.training(loss_op)

  # Add the Op to compare the logits to the labels during evaluation on the complete test set
  #labels_predict = model.make_predictions(logits)
  eval_op = model.evaluation_stats(logits, labels_placeholder, 1)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver(sharded=True)

  # Run the Op to initialize the variables.
  tf.global_variables_initializer().run(session=sess)

  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

  ckpt = tf.train.get_checkpoint_state(logs_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print('Checkpoint found! Will train from here')
    saver.restore(sess, ckpt.model_checkpoint_path)
    offset_step = sess.run(global_step_tensor)
    print('Resuming from step ', offset_step)
  else:
    print('No checkpoint found... will train from scratch')
    offset_step = 0


  # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
  config = projector.ProjectorConfig()

  # You can add multiple embeddings. Here we add only one.
  embedding_features = config.embeddings.add()
  #embedding_features.tensor_name = embedding_var.name

  # Link this tensor to its metadata file (e.g. labels).
  embedding_features.metadata_path = os.path.join('/media/sf_shared/Tutorial/smarter/data/', 'classes.tsv')

  # Saves a configuration file that TensorBoard will read during startup.
  projector.visualize_embeddings(summary_writer, config)

   # And then after everything is built, start the training loop.
  for step in xrange(max_steps):
    global_step = step + offset_step
    start_time = time.time()

    # Fill a feed dictionary with the actual set of features and labels for this particular training step.
    feed_dict,_ = common.fill_feed_dict(data_sets.train, features_placeholder, labels_placeholder, batch_size)

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, __, loss_value, summary_str = sess.run([eval_op, train_op, loss_op, summary_op], feed_dict=feed_dict)

    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
      # Print status to stdout.
      duration = time.time() - start_time
      print('Step %d: loss = %.2f (%.3f sec)' % (global_step, loss_value, duration))
      # Update the events file.
      #summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, global_step)

    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
      sess.run(global_step_tensor.assign(global_step))
      saver.save(sess, logs_dir + 'model.ckpt', global_step=global_step)

      # Evaluate against the training set.
      #print('Training Data Eval:')
      #common.do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.train)
      # Evaluate against the validation set.
      #print('Validation Data Eval:')
      #common.do_eval(sess, eval_correct, features_placeholder, labels_placeholder, data_sets.validation)
      # Evaluate against the test set.
      print('Test Data Eval:')
      common.do_eval(sess, eval_op, features_placeholder, labels_placeholder, data_sets.test)

      #export4serving(sess, saver, features_placeholder, labels_placeholder)

