# Mostly from tf docs, used to test computer

import time
start_time = time.time()
import input_data
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Since we're using ReLU neurons, best to initialize weights
# with slight positive bias to avoid 'dead neurons'
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution with stride of 1, zero padding so output same as input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Standard max pooling over 2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

#####################################################
# First Conv layer
#####################################################
"""
We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
"""

# Placeholders for x and y values
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Conv of 32 features for each 5x5 patch, 1 input channel
W_conv1 = weight_variable([5, 5, 1, 32]) # conv weights
b_conv1 = bias_variable([32]) # conv biases

# Reshape x to 4d tensor
# 28x28 image, 1 color channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with weight tensor, add bias, apply ReLU and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#####################################################
# Second Conv layer
#####################################################
# Deep network, we stack more layers
# This layer has 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#####################################################
# Densely connected layer
#####################################################
"""
Now that the image size has been reduced to 7x7 (each previous convolution halved the sze), we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#####################################################
# Dropout - regularization
#####################################################
# Dropout to reduce overfitting
# As placeholder for probability of neuron dropout
# Dropout on while training, off for test
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#####################################################
# Readout layer
#####################################################
# Softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#####################################################
# Loss and Backpropagation
#####################################################
# Same loss function as NN
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# We use ADAM optimizer instead of gradient descient, more advanced
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#####################################################
# Train and Evaluate
#####################################################

# Set some evaluation metrics
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Train
sess = tf.Session()
sess.run(tf.initialize_all_variables()) # init all vars
with sess.as_default():
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Test!
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print("--- %s seconds ---" % (time.time() - start_time))
