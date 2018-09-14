import tensorflow as tf
import numpy as np
from training import parameters as param
from training import guru_model as model
from six.moves import xrange  # pylint: disable=redefined-builtin
import time

def run_training(file):
    if(file=='pessimist'):
        filename=param.pessimist_filename
    if(file=='real'):
        filename=param.real_filename
    data = np.loadtxt(open(filename, "rb"), delimiter=",", dtype='float32')
    data_sets = model.DataSets()
    data_sets.train = model.DataSet(data[:,0:param.features_size], 
    data[:,param.features_size:param.features_size+param.labels_size])

    # Config to turn on JIT compilation
    tfconfig = tf.ConfigProto()
    sess = tf.Session(config=tfconfig)
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # Creates a variable to hold the global_step.
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    # Placeholders
    features_placeholder = tf.placeholder(tf.float32, None)
    labels_placeholder = tf.placeholder(tf.float32, None)
    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(features_placeholder)
    # Add to the Graph the Ops for loss calculation.
    loss_op = model.loss(logits, labels_placeholder)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = model.training(loss_op)
    # Add the Op to compare the logits to the labels during evaluation on the complete test set
    eval_op = model.evaluation_stats(logits, labels_placeholder)
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(sharded=True)
    # Run the Op to initialize the variables.
    #tf.global_variables_initializer().run(session=sess)
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(param.logs_dir, sess.graph)
    # Function to initialize custom variables after restore checkpoint
    running_vars_initializer = model.initializeCustomVariables()

    ckpt = tf.train.get_checkpoint_state(param.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Checkpoint found! Will train from here')
        sess.run(running_vars_initializer)
        saver.restore(sess, ckpt.model_checkpoint_path)
        offset_step = sess.run(global_step_tensor)
        print('Resuming from step ', offset_step)
    else:
        print('No checkpoint found... will train from scratch')
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        offset_step = 0

    # Start run loop
    max_steps = param.max_steps
    for step in xrange(max_steps):
        global_step = step + offset_step
        start_time = time.time()
        features_feed, labels_feed = data_sets.train.next_batch(param.batch_size)
        # Fill a feed dictionary with the actual set of features and labels for this particular training step.
        feed_dict = model.fill_feed_dict(features_feed, labels_feed,
                                features_placeholder, labels_placeholder)
        metrics, __, loss_value, summary_str = sess.run([eval_op, train_op, loss_op, summary_op], feed_dict=feed_dict)
        #accuracy = sess.run(metrics[0], feed_dict=feed_dict) #accuracy
        #recall = sess.run(metrics[1], feed_dict=feed_dict) #recall
        #precision = sess.run(metrics[2], feed_dict=feed_dict) #precision
        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            duration = time.time() - start_time
            # Update the events file.
            summary_writer.add_summary(summary_str, global_step)
        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
            print('Step %d: loss = %.2f (%.3f sec)' % (global_step, loss_value, duration))
            print('Accuracy = %.3f; Recall = %.3f; Precision = %.3f' % (metrics[0], metrics[1], metrics[2]))
            sess.run(global_step_tensor.assign(global_step))
            saver.save(sess, param.logs_dir + 'model.ckpt', global_step=global_step)
        # Export model and evaluate the model periodically.
        if (step + 1) == max_steps:
            tf.saved_model.simple_save(sess,param.logs_dir + 'model_export', 
                {"features": features_feed}, {"binary_classif": labels_feed})