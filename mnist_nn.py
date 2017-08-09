# partially follows tut in tensor flow doc, MNIST for ML beg
# https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html
# I expanded the logistic model into a NN with 1 hidden layer
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

# x is a placeholder for data to be loaded later
# None because we don't know how many values (images) will be loaded
# in 1st dim. 784 because flattened 28*28 image
x = tf.placeholder(tf.float32, [None, 784])
num_labels = 10

# Hidden layer variables
h_units = 40
W_h = tf.Variable(tf.truncated_normal([784, h_units], stddev=0.1))
b_h = tf.Variable(tf.zeros([h_units]))
hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_h), b_h))

# Hidden-to-output parameters
W_o = tf.Variable(tf.truncated_normal([h_units, num_labels], stddev=0.1))
b_o = tf.Variable(tf.zeros([num_labels]))
logits = tf.matmul(hidden_1, W_o) + b_o
y = tf.nn.softmax(logits)

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

# Get the accuracy on test data, compare argmax of pred and correct
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# We cast the list of booleans into floats and get the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train in batches of 100, 1000 times
# Small random batches is stochastic training, or rather stochastic
# gradient descent in our case
epochs = 20

for r in range(epochs):
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # when we run we feed batch of x/y to train_step
        # this will replace the placeholder with our data
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print 'epoch ', r, ' : ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

# Print the accuracy of our test
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

