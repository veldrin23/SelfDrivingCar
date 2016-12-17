# Load pickled data
import pickle
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import time
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from scipy.misc import imresize
import random

training_file = '/home/veldrin/CarND-Traffic-Signs/train.p'
testing_file = '/home/veldrin/CarND-Traffic-Signs/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
labels = pd.read_csv('/home/veldrin/CarND-Traffic-Signs/signnames.csv')['SignName']
# Free up memory
del train, test


a = random.sample(range(X_train.shape[0]), 10)

for b in a:
    # plt.imshow(X_train[b])
    # plt.show()
    mpimg.imsave('/home/veldrin/CarND-Traffic-Signs/own_images/' + str(b) + '.jpg',X_train[b])
    # mpimg.imsave()

n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:3]
n_classes = pd.read_csv('/home/veldrin/CarND-Traffic-Signs/signnames.csv')['ClassId'].__len__()


# Change labels to one-hot matrices
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

y_train = dense_to_one_hot(y_train, num_classes=43)
y_test = dense_to_one_hot(y_test, num_classes=43)

# Normalize
def normalize(img):
    return cv2.normalize(img, None, 0.0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sharpen
def sharpen(img):
    kernel_size = 0
    g_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 3)
    img_copy = np.copy(img)
    return cv2.addWeighted(img_copy, 1.5, g_img, -0.5, 0, g_img)

# Combined
def process_image(img):
    gray = grayscale(img)
    sharp_norm_gray = sharpen(gray )
    norm_gray = normalize(sharp_norm_gray)
    return norm_gray

X_train = np.array([process_image(image) for image in X_train], dtype=np.float32)
X_test = np.array([process_image(image) for image in X_test], dtype=np.float32)

# Split part of test set for validation

X_train_features, X_valid_features, y_train_labels, y_valid_labels = train_test_split(
    X_train,
    y_train,
    test_size=0.10,
    random_state=42)

learning_rate = 0.001
training_epochs = 15
batch_size = 64
dropout = 0.65

layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'layer_3': 54,
    'out': 120
}

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 1, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'out': tf.Variable(tf.zeros(layer_width['out']))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def ConvNet(x, weights = weights, biases = biases):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 32, 32, 1))

    x /= 255.

    # Layer 1: relu, dropout, maxpool
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = tf.nn.dropout(conv1, dropout)
    conv1 = maxpool2d(conv1)
    # conv1 = tf.nn.dropout(conv1, dropout)
    # Layer 2
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    fc1 = flatten(conv2)

    fc1_shape = (fc1.get_shape().as_list()[-1], layer_width['out'])
    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
    fc1_b = tf.Variable(tf.zeros(layer_width['out']))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(layer_width['out'], n_classes)))
    fc2_b = tf.Variable(tf.zeros(n_classes))
    return tf.matmul(fc1, fc2_W) + fc2_b


# input data consists of 32x32, grayscale min-max scaled images.
x = tf.placeholder(tf.float32, (None, 32, 32))
# Classify over 43 digits 0-42.
y = tf.placeholder(tf.float32, (None, n_classes))
# Create the LeNet.
logits = ConvNet(x)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer() # after many moons of fooling around, I found that SGD is not converging at a pace that suits humans' lifespan
train_op = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()
#
def next_batch(x, y, batch_index, batch_size):
    start = batch_index + 1
    end = start + batch_size
    return x[start:end, :], y[start:end]


def eval_data(features, labels):
    """
    Given features and lables as input, returns the loss and accuracy.
    """
    steps_per_epoch = len(features) // batch_size
    num_examples = steps_per_epoch * batch_size

    total_acc, total_loss = 0, 0
    batch_index = 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = next_batch(features, labels, batch_index, batch_size)
        loss, acc = sess.run([cost, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
        batch_index += batch_size
    return total_loss / num_examples, total_acc / num_examples


if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = len(X_train_features) // batch_size
        num_examples = steps_per_epoch * batch_size
        print("begin first epoch loss calc =", time.strftime("%c"))

        # Training cycle
        for i in range(training_epochs):
            batch_index = 0
            for step in range(steps_per_epoch):
                batch_x, batch_y = next_batch(X_train_features, y_train_labels,
                                              batch_index, batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                batch_index += batch_size
            #
            # val_loss, val_acc = eval_data(valid_features,valid_labels)
            val_loss, val_acc = eval_data(X_valid_features, y_valid_labels)
            print(time.strftime("%X"), "{}".format(i + 1),
                  "loss = {}".format(val_loss), "accuracy = {}".format(val_acc))

        # Evaluate on the test data
        # test_loss, test_acc = eval_data(X_test,y_test)
        test_loss, test_acc = eval_data(X_test, y_test)
        print("Test loss = {}".format(test_loss),
              "Test accuracy = {}".format(test_acc))

init_op = tf.initialize_all_variables()
prediction = tf.nn.softmax(logits)

test_dir = '/home/veldrin/CarND-Traffic-Signs/own_images'
test_images = os.listdir(test_dir)


fig = plt.figure()
j = 1
for i in test_images:
    img_in = mpimg.imread(test_dir + '/' + i)
    test_image = imresize(process_image(img_in), (32, 32))
    with tf.Session() as session:
        session.run(init_op)
        pred = session.run(prediction, feed_dict = {x: [test_image]})

    fig.add_subplot(5,2,j)
    plt.imshow(img_in)
    plt.title(labels.values[pred[0] == 1][0])
    plt.axis('off')
    j += 1

plt.show()
