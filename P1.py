# Load pickled data
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import random

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

plt.rcParams['image.cmap'] = 'gray'


# TODO: Fill this in based on where you saved the training and testing data

training_file = '/home/veldrin/CarND-Traffic-Signs/train.p'
testing_file = '/home/veldrin/CarND-Traffic-Signs/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# y_train, y_test = tf.expand_dims(y_train, 1) , tf.expand_dims(y_test, 1)



### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]
# 32x32

# TODO: How many unique classes/labels there are in the dataset.
n_classes = pd.read_csv('/home/veldrin/CarND-Traffic-Signs/signnames.csv')['ClassId'].__len__()


# print(tf.one_hot(y_train[1], n_classes))




# random15 = np.random.choice(np.arange(X_train.shape[0]), 15)
#
# fig = plt.figure()
#
# for i in range(15):
#     fig.add_subplot(3, 5, i+1)
#     plt.imshow(X_train[random15[i]])our
#     plt.axis('off')
#     plt.title('label: %s' %(y_train[random15[i]]))
#
#
#
# plt.figure(2)
# weights_train = np.ones_like(y_train)/len(y_train)
# plt.hist(y_train, bins = max(y_train), color='dodgerblue', weights=weights_train, alpha = 0.5)
#
# weights_test = np.ones_like(y_test)/len(y_test)
# plt.hist(y_test+.2, bins = max(y_test), color='firebrick', weights=weights_test, alpha = 0.5)
#
# plt.title('Frequency of labels in training data (blue) and test data (red)')



def normalize(img):
    return cv2.normalize(img, None, 0.0, 0.1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





def shuffle_data(x, y):
    data_lenth = x.shape[0]
    shuffled_ints = np.arange(data_lenth)
    random.shuffle(shuffled_ints)
    return x[shuffled_ints], y[shuffled_ints]

x2, y2 = shuffle_data(X_train, y_train)


X_train2 = np.array([normalize(image) for image in x2], dtype=np.float32)
X_train2 = np.array([grayscale(image) for image in X_train2], dtype=np.float32)
X_train2 = X_train2[..., np.newaxis] # reshapes the image data; grayscale reduces dimensions

X_test2 = np.array([normalize(image) for image in X_test], dtype=np.float32)
X_test2 = np.array([grayscale(image) for image in X_test2], dtype=np.float32)
X_test2 = X_test2[ ..., np.newaxis] # reshapes the image data; grayscale reduces dimensions


# TF doesn't like np that much (yet, apparently)
# print(y_train.shape, y_test.shape)
y_train = dense_to_one_hot(y_train, n_classes)
y_test = dense_to_one_hot(y_test, n_classes)
# print(y_train.shape, y_test.shape)


# Parameters
learning_rate = 0.001
batch_size = 128
training_epochs = 15
depth = X_train2.shape[3]

# image shape 32x32

# plt.show()
layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, depth, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [4*4*128, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Layer 1 -
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2 -
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3 -
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    # Fully connected layer - 4*4*128 to 512
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(
        conv3,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction - 512 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



# tf Graph input
x = tf.placeholder("float", [None, 32, 32, depth])
y = tf.placeholder("float", [None, n_classes])

logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()
#
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(n_test/batch_size)
        # Loop over all batches
        for i in range(total_batch):

            batch_x, batch_y = X_train2[(i)*batch_size:(i+1)*batch_size], y_train[(i)*batch_size:(i+1)*batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%02d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        "Accuracy:",

        accuracy.eval({x: X_test2, y: y_test}))
