import tensorflow as tf
import numpy as np

import pandas
print(pandas.__version__)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt
print("Shape of mnist.train.images {0}".format(mnist.train.images.shape))
print("Shape of mnist.test.images {0}".format(mnist.test.images.shape))
print("Shape of mnist.validation.images {0}".format(mnist.validation.images.shape))

print("Image:")
plt.imshow(mnist.test.images[5].reshape(28,28), cmap='gray')
plt.show()
print("Label: ")
print(mnist.test.labels[5])

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


prediction = tf.argmax(y,1)
labels = tf.argmax(y_, 1)
predicted_results = prediction.eval(feed_dict={x: mnist.test.images}, session=sess)
real_results = labels.eval(feed_dict={y_: mnist.test.labels}, session=sess)
print("ImageId, Label")
for i in range(10):
    print("{0} {1} {2}". format(i, predicted_results[i], real_results[i]))
    print("Image:")
    plt.imshow(mnist.test.images[i].reshape(28,28), cmap='gray')
    plt.show()
    print("Label: ")
    print(mnist.test.labels[i])
