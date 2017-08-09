# follows tut in tensor flow doc, MNIST for ML beg
# https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html
import input_data
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


"""
Model definition

We want to implement as simple NN to predict the MNIST dataset.
Basically, we want to implement the following formula:
    y = softmax(Wx + b)
where:
    W is a weight matrix of [784,10]
    x is the input vector [784]
    b is the bias vector [10]
    y is our label of 10 numbers, mapped to 0-9 in one-hot encoding
"""

# x is a placeholder, not a value
# None because we don't know how many values (images) will be loaded
# in 1st dim. 784 because flattened 28*28 image
x = tf.placeholder(tf.float32, [None, 784])

# A variable is a modifiable tensor that lives in tf's graph
# Model parameters should be Variables
# We create vars for weights W and biases b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)


"""
LOSS function

We use the cross-entropy formula to measure loss:
    H_{y'}(y) = -\sum_i y'_i \log(y_i)
where:
    y is our predicted probability distribution
    y' is the true distribution
It is the negative sum, for all i predictions, of actual*log(pred)
"""

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Backpropagation with gradient descent, learning rate of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize all variables and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train in batches of 100, 1000 times
# Small random batches is stochastic training, or rather stochastic
# gradient descent in our case
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # when we run we feed batch of x/y to train_step
    # this will replace the placeholder with our data
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Get the accuracy on test data, compare argmax of pred and correct
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# We cast the list of booleans into floats and get the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Print the accuracy of our test
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

