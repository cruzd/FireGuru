
# coding: utf-8

# # Generate data to feed training

# In[1]:


import tensorflow as tf
import numpy as np

featuresSize = 18
labelsSize = 2
filename = '/data/prepared/teste.txt'

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

teste = np.loadtxt(open(filename, "rb"), delimiter=",", dtype='float32')
data_sets = DataSets()
data_sets.train = DataSet(teste[:,0:featuresSize], teste[:,featuresSize:featuresSize+labelsSize])


# # Model defining

# In[2]:


slim = tf.contrib.slim

batch_size = 1000
learning_rate = 0.01
logs_dir = './'
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
    # Create the feed_dict for the placeholders filled with the next `batch size` examples.
    feed_dict = {
        features_pl: features,
        labels_pl: labels,
    }
    return feed_dict

def variable_summaries(var):
  #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
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
        optimizer = tf.train.AdamOptimizer(learning_rate)
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

def evaluation_stats(logits, labels, k=1):
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.f.float32))
    predictions = make_predictions(logits)
    recall, recall_op = tf.metrics.recall(tf.argmax(labels,1), predictions)
    precision, precision_op = tf.metrics.precision(tf.argmax(labels,1), predictions)
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(labels,1), predictions)
    #labels = tf.squeeze(tf.argmax(labels,1))
    #recall_op = slim.metrics.streaming_precision(labels, predictions)
    #precision_op = slim.metrics.streaming_precision(labels, predictions)
    #accuracy_op = slim.metrics.streaming_precision(labels, predictions)
    tf.summary.scalar('precision', precision_op*100)
    tf.summary.scalar('recall', recall_op*100)
    tf.summary.scalar('accuracy', accuracy_op*100)
    # Return the number of true entries.
    return [accuracy_op, recall_op, precision_op]

def do_eval(sess, eval_correct, features_placeholder, labels_placeholder, test_features, test_labels):
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.

    steps_per_epoch = len(test_features) // batch_size
    num_examples = steps_per_epoch * batch_size
    print('Batch size is ', batch_size, 'Steps per epoch are ', steps_per_epoch, 'Samples: ', num_examples)

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(test_features, test_labels, features_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples

    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

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
logits = inference(features_placeholder)
# Add to the Graph the Ops for loss calculation.
loss_op = loss(logits, labels_placeholder)
# Add to the Graph the Ops that calculate and apply gradients.
train_op = training(loss_op)
# Add the Op to compare the logits to the labels during evaluation on the complete test set
eval_op = evaluation_stats(logits, labels_placeholder, 1)
# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()
# Create a saver for writing training checkpoints.
saver = tf.train.Saver(sharded=True)
# Run the Op to initialize the variables.
tf.global_variables_initializer().run(session=sess)
# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter('', sess.graph)

predictions = tf.argmax(logits, 1)
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels_placeholder,1), predictions=predictions)
rec, rec_op = tf.metrics.recall(labels=tf.argmax(labels_placeholder,1), predictions=predictions)
pre, pre_op = tf.metrics.precision(labels=tf.argmax(labels_placeholder,1), predictions=predictions)

# Create a saver for writing training checkpoints.
#saver = tf.train.Saver(sharded=True)
#ckpt = tf.train.get_checkpoint_state(logs_dir)
#if ckpt and ckpt.model_checkpoint_path:
    #print('Checkpoint found! Will train from here')
    #saver.restore(sess, ckpt.model_checkpoint_path)
    #offset_step = sess.run(global_step_tensor)
    #print('Resuming from step ', offset_step)
#else:
    #print('No checkpoint found... will train from scratch')
    #offset_step = 0


# # Training

# In[ ]:


from six.moves import xrange  # pylint: disable=redefined-builtin
import time
max_steps = 1500000000
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
for step in xrange(max_steps):
    #global_step = step + offset_step
    global_step = step
    start_time = time.time()
    features_feed, labels_feed = data_sets.train.next_batch(batch_size)
    # Fill a feed dictionary with the actual set of features and labels for this particular training step.
    feed_dict = fill_feed_dict(features_feed, labels_feed,
                               features_placeholder, labels_placeholder)

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    metrics, __, loss_value, summary_str = sess.run([eval_op,
                                                                         train_op, loss_op, summary_op], feed_dict=feed_dict)
    accuracy = sess.run(acc_op, feed_dict=feed_dict) #accuracy
    recall = sess.run(rec_op, feed_dict=feed_dict) #recall
    precision = sess.run(pre_op, feed_dict=feed_dict) #precision
    #accuracy=metrics[0]
    #recall=metrics[1]
    #precision=metrics[2]
    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
        # Print status to stdout.
        duration = time.time() - start_time
        # Update the events file.
        summary_writer.add_summary(summary_str, global_step)
    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
        print('Step %d: loss = %.2f (%.3f sec)' % (global_step, loss_value, duration))
        print('Accuracy = %.3f; Recall = %.3f; Precision = %.3f' % (accuracy, recall, precision))
        sess.run(global_step_tensor.assign(global_step))
        #saver.save(sess, logs_dir + 'model.ckpt', global_step=global_step)
        #saver.save(sess, 'model.ckpt', global_step=global_step)
        #print('Test Data Eval:')
        #do_eval(sess, eval_op, features_placeholder, labels_placeholder, teX, teY)

