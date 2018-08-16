import tensorflow as tf
import numpy as np
from training import parameters as param
slim = tf.contrib.slim

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
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._num_examples
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
            end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]
    
class DataSets(object):
    pass

# Model specific stuff

def get_num_classes():
    return 2
def get_num_hidden1():
    return 8
def get_num_features():
    return 18
def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def fill_feed_dict(features, labels, features_pl, labels_pl):
    feed_dict = {
        features_pl: features,
        labels_pl: labels,
    }
    return feed_dict

#Attach a lot of summaries to a Tensor (for TensorBoard visualization).
def variable_summaries(var):
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
    layer_1 = nn_layer(features, get_num_features(), get_num_hidden1(), 'layer1', act=tf.nn.relu)
    logits = nn_layer(layer_1, get_num_hidden1(), get_num_classes(), 'out', act=tf.identity)
    return logits

def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=10000)
        #diff = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def training(loss):
    with tf.name_scope('train'):
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(param.learning_rate)
        # Create a variable to track the global step.
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss) #, global_step=global_step)
    return train_op

def make_predictions(logits):
    # Get the predicted positions, index
    predicted = tf.argmax(logits, 1)
    return predicted

def evaluation_stats(logits, labels):
    predictions = make_predictions(logits)
    recall, recall_op = tf.metrics.recall(tf.argmax(labels,1), predictions)
    precision, precision_op = tf.metrics.precision(tf.argmax(labels,1), predictions)
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(labels,1), predictions)
    tf.summary.scalar('precision', precision_op*100)
    tf.summary.scalar('recall', recall_op*100)
    tf.summary.scalar('accuracy', accuracy_op*100)
    return [accuracy_op, recall_op, precision_op]

def do_eval(sess, eval_correct, features_placeholder, labels_placeholder, test_features, test_labels):
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    batch_size = param.batch_size
    steps_per_epoch = len(test_features) // batch_size
    num_examples = steps_per_epoch * batch_size
    print('Batch size is ', batch_size, 'Steps per epoch are ', steps_per_epoch, 'Samples: ', num_examples)

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(test_features, test_labels, features_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples

    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))



def set_model():
    # Config to turn on JIT compilation
    tfconfig = tf.ConfigProto()
    sess = tf.Session(config=tfconfig)
    return sess