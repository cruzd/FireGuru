from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn import datasets, metrics
import numpy as np


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
    features_placeholder = tf.placeholder(tf.float32, [batch_size, None])  # num_features
    labels_placeholder = tf.placeholder(tf.int64, [batch_size])
    return features_placeholder, labels_placeholder


def fill_feed_dict(data_set, features_pl, labels_pl, batch_size):
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
    features_feed, labels_feed = data_set.next_batch(batch_size)

    # print data to see it's ok
    # print('features shape: ', features_feed.shape)
    # print('labels shape: ', labels_feed.shape)

    # for i in xrange(2):
    #  print(features_feed[i], labels_feed[i])

    feed_dict = {
        features_pl: features_feed,
        labels_pl: labels_feed,
    }
    return feed_dict, labels_feed


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

    batch_size = features_placeholder.get_shape()[0].value
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    print('Batch size is ', batch_size, 'Steps per epoch are ', steps_per_epoch, 'Samples: ', num_examples)

    for step in xrange(steps_per_epoch):
        feed_dict, _ = fill_feed_dict(data_set, features_placeholder, labels_placeholder, batch_size)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples

    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def do_stats(sess, labels_predict, features_placeholder, labels_placeholder, data_set):
    """
    Runs one evaluation against the data set
    Args:
      sess: The session in which the model has been trained.
      labels_predict: The Tensor that returns the predicted labels
      labels_placeholder: The labels placeholder.
      data_set: The set of features and labels to evaluate, from input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    bulk_size = features_placeholder.get_shape()[0].value

    feed_dict, labels_true = fill_feed_dict(data_set, features_placeholder, labels_placeholder, bulk_size)
    predicted_labels = sess.run(labels_predict, feed_dict=feed_dict)

    print ('Precision:', metrics.precision_score(labels_true, predicted_labels, average='weighted'))
    print ('Recall:', metrics.recall_score(labels_true, predicted_labels, average='weighted'))
    print ('F1_score:', metrics.f1_score(labels_true, predicted_labels, average='weighted'))
    print ('Confusion Matrix')
    outfile = open('confusion_matrix.out', 'w')
    np.set_printoptions(threshold=np.inf, linewidth=3000)
    cm = metrics.confusion_matrix(labels_true, predicted_labels, labels=list(xrange(0, 180, 1)))
    ###print ('Shape:', cm.shape())
    print (cm, file=outfile)
    # fpr, tpr, tresholds = metrics.roc_curve(labels_true, labels_predict)