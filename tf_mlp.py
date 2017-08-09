import tensorflow as tf
import numpy as np
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer as glorot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from data import disp_img

size_train = mnist.train.num_examples # should be 55k
size_val = mnist.validation.num_examples # should be 5k
size_test = mnist.test.num_examples # should be 10k
input_dim = 784
num_classes = 10
drop_rate = 0.2
keep_prob = 0.8
batch_size = 100
batch_val_size = 100
nb_epoch = 30


data_type = tf.float32
def dense(x, in_dim, out_dim, scope, act=None, drop=1):
  """
  Fully connected layer builder
  Args:
    in_dim  - dimensions of input
    out_dim - number of hidden unist
    scope   - scope name
    act     - activation function, default no activation (i.e. linear)
    drop    - the dropout keep rate, default is 1 (no dropout)
  """
  with tf.variable_scope(scope):
    weights = tf.get_variable("weights", shape=[in_dim, out_dim],
              dtype=data_type, initializer=glorot())
    biases = tf.get_variable("biases", out_dim,
              dtype=data_type, initializer=tf.constant_initializer(0.0))
    # Matrix multiply, input X weights
    h = tf.matmul(x,weights) + biases
    # Post activation
    if act:
      h = act(h)
    # Dropout
    if drop < 1:
      h = tf.nn.dropout(h, drop)
    return h

# Validation batch check
assert size_val % batch_val_size == 0

# Declare placeholders: variables you feed during training
x = tf.placeholder(tf.float32, [None, 784])
y_target = tf.placeholder(tf.float32, [None, 10])

# Hidden layers
units = 3000
h_1 = dense(x, input_dim, units, "W_1", act=tf.nn.relu, drop=keep_prob)
h_2 = dense(h_1, units, units, "W_2", act=tf.nn.relu, drop=keep_prob)
y = dense(h_2, units, num_classes, "out")

# Configure optimization
loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
optimizer_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

# Configure evaluation
# correct_prediction is a list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
# cast to the booleans to float and take the average, i.e. accuracy
acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def eval_validation(sess):
  loss_ls = []
  acc_ls = []
  steps = size_val // batch_val_size
  for _ in range(steps):
    batch_xs, batch_ys = mnist.validation.next_batch(100)
    feed_dict={x: batch_xs, y_target: batch_ys}
    fetch = [optimizer_op, loss_op, acc_op]
    _, loss, acc = sess.run(fetch, feed_dict)
    loss_ls.append(loss)
    acc_ls.append(acc)
  loss = np.mean(loss_ls)
  acc = np.mean(acc_ls)
  print(' || val loss: {:.4f} | accuracy: {:.4f}'.format(
        loss, acc) )

# Context manager for convenience
# Uses default graph
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  epochs_completed = 0
  print('*'*79)
  print('TRAINING')
  print('*'*79)
  t1 = datetime.now()
  while mnist.train.epochs_completed < nb_epoch:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict={x: batch_xs, y_target: batch_ys}
    fetch = [optimizer_op, loss_op, acc_op]
    _, loss, acc = sess.run(fetch, feed_dict)
    t2 = datetime.now()
    diff_t = (t2 - t1).total_seconds()
    print('epoch: {:2.0f} time: {:2.1f} | loss: {:.4f} | accuracy: {:.4f}'.format(
        epochs_completed, diff_t, loss, acc), end='\r')
    if mnist.train.epochs_completed > epochs_completed:
      t2 = datetime.now()
      diff_t = (t2 - t1).total_seconds()
      epochs_completed = mnist.train.epochs_completed
      print('epoch: {:2.0f} time: {:2.1f} | loss: {:.4f} | accuracy: {:.4f}'.format(
        epochs_completed, diff_t, loss, acc), end="")
      eval_validation(sess)
      t1 = datetime.now()

