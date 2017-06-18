# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.utils import shuffle
import time
# Fill this in based on where you saved the training and testing data

training_file = './train.p'
validation_file='./valid.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


X_train = X_train.astype(float)
X_train = (X_train - 128)/128
# %%

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.


# Visualizations will be shown in the notebook.
#%matplotlib inline

#index = random.randint(0, len(X_train))
#image = X_train[index].squeeze()
#show one example
#plt.figure(figsize=(1,1))
#plt.imshow(image)

#Count of each class
count = np.bincount(y_train)
ii = np.nonzero(count)[0]
freq = zip(ii, count[ii])
print('Count of each class in list')
print([i for i in freq])

#print('Class: ',y_train[index])

# %%

def fc(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    epsilon = 1e-3 #batchnorm parameter
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    F_W = tf.Variable(tf.truncated_normal([5,5,3,6], mean=mu, stddev=sigma))
    F_b = tf.Variable(tf.zeros(6))

    
    strides = [1, 1, 1, 1]
    conv1 = tf.nn.conv2d(x, F_W, strides, 'VALID') + F_b
    print(conv1)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # Layer 2: Convolutional. Output = 10x10x16.
    
    F_W2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev=sigma))
    F_b2 = tf.Variable(tf.zeros(16))
    strides2 = [1,1,1,1]
    conv2 = tf.nn.conv2d(conv1, F_W2, strides, 'VALID') + F_b2
    # Activation.
    
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Flatten. Input = 5x5x16. Output = 400.
    F_W3 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    F_b3 = tf.Variable(tf.zeros(120))
    fc1 = tf.reshape(conv2, [-1, F_W3.get_shape().as_list()[0]])
    
    fc1 = fc_bn_relu(fc1, 120, phase, 'fc1')
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    #fc1 = tf.matmul(fc1,F_W3)
    
    #Batch norm
    #batch_mean1, batch_var1 = tf.nn.moments(fc1, [0])
    #z1_hat = (fc1 - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
    #scale1 = tf.Variable(tf.ones([120]))
    #beta1 = tf.Variable(tf.zeros([120]))
    
    #BN1 = scale1 * z1_hat + beta1
    
    # Activation.
    #BN1 = tf.nn.relu(BN1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    F_W4 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev = sigma))
    F_b4 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,F_W4), F_b4)
    
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    F_W5= tf.Variable(tf.truncated_normal([84, n_classes], mean=mu, stddev = sigma))
    F_b5 = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2,F_W5), F_b5)
    
    return logits

# %%
def VGG(x):
    mu = 0
    sigma = 0.1
    strides = [1,1,1,1]
    # Conv Layer 1_1. Input = 32x32x3. Output = 32x32x64.
    F_W = tf.Variable(tf.truncated_normal([3,3,3,32], mean=mu, stddev=sigma))
    F_b = tf.Variable(tf.zeros(32))


    conv1_1 = tf.nn.conv2d(x, F_W, strides, 'SAME') + F_b
    conv1_1 = tf.nn.relu(conv1_1)
    
    # Conv Layer 1_2: Input 32x32x64. Output = 32x32x64
    F_W2 = tf.Variable(tf.truncated_normal([3,3,32,32], mean=mu, stddev=sigma))
    F_b2 = tf.Variable(tf.zeros(32))

    conv1_2 = tf.nn.conv2d(conv1_1, F_W2, strides, 'SAME') + F_b2
    conv1_2 = tf.nn.relu(conv1_2)
    
    # Pooling 1. Input = 32x32x64. Output = 16x16x64.
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Conv Layer 2_1. Input = 16x16x64. Output = 16x16x128.
    F_W3 = tf.Variable(tf.truncated_normal([3,3,32,64], mean=mu, stddev=sigma))
    F_b3 = tf.Variable(tf.zeros(64))

    conv2_1 = tf.nn.conv2d(pool1, F_W3, strides, 'SAME') + F_b3
    conv2_1 = tf.nn.relu(conv2_1)
    
    # Conv Layer 2_2. Input = 16x16x128. Output = 16x16x128.
    F_W4 = tf.Variable(tf.truncated_normal([3,3,64,64], mean=mu, stddev=sigma))
    F_b4 = tf.Variable(tf.zeros(64))

    conv2_2 = tf.nn.conv2d(conv2_1, F_W4, strides, 'SAME') + F_b4
    conv2_2 = tf.nn.relu(conv2_2)
    
    # Pooling 2. Input = 16x16x128. Output = 8x8x128.
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    """
    # Conv Layer 3_1. Input = 8x8x128. Output = 8x8x256.
    F_W5 = tf.Variable(tf.truncated_normal([3,3,128,256], mean=mu, stddev=sigma))
    F_b5 = tf.Variable(tf.zeros(256))

    conv3_1 = tf.nn.conv2d(pool2, F_W5, strides, 'SAME') + F_b5
    conv3_1 = tf.nn.relu(conv3_1)
    
    # Conv Layer 3_2. Input = 8x8x256. Output = 8x8x256.
    F_W6 = tf.Variable(tf.truncated_normal([3,3,256,256], mean=mu, stddev=sigma))
    F_b6 = tf.Variable(tf.zeros(256))

    conv3_2 = tf.nn.conv2d(conv3_1, F_W6, strides, 'SAME') + F_b6
    conv3_2 = tf.nn.relu(conv3_2)
    
    # Pooling 3. Input = 8x8x256. Output = 4x4x256.
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    """
    
    # Flatten. Input = 8x8x128 = 8192. Output = 128
    F_W7 = tf.Variable(tf.truncated_normal([4096, 128], mean=mu, stddev=sigma))
    F_b7 = tf.Variable(tf.zeros(128))
    fc1 = tf.reshape(pool2, [-1, F_W7.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, F_W7), F_b7)
    #fc1 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=phase, scope='bn1')
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    
    # Fully Connected Layer 2. Input = 128. Output = 43
    F_W8 = tf.Variable(tf.truncated_normal([128, n_classes], mean=mu, stddev=sigma))
    F_b8 = tf.Variable(tf.zeros(n_classes))
    fc2 = tf.add(tf.matmul(fc1, F_W8), F_b8)
    #fc2 = tf.contrib.layers.batch_norm(fc2, center=True, scale=True, is_training=phase, scope='bn2')
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    """
    # Fully Connected Layer 3. Input = 43. Output = 43    
    F_W9 = tf.Variable(tf.truncated_normal([128, n_classes], mean=mu, stddev=sigma))
    F_b9 = tf.Variable(tf.zeros(n_classes))
    fc3 = tf.add(tf.matmul(fc2, F_W9), F_b9)
    
    """
    F_W10 = tf.Variable(tf.truncated_normal([n_classes, n_classes], mean=mu, stddev=sigma))
    F_b10 = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2, F_W10), F_b10)
    
    return logits


# %%
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)
phase = tf.placeholder(tf.bool)

learning_rate = 5e-4
EPOCHS = 100
BATCH_SIZE = 128

#logits = LeNet(x)
logits = VGG(x)

# For decaying learning rate
global_step = tf.Variable(0, trainable=False)
decaying_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(decaying_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1, phase: 0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# %%

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    starttime = time.time()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, phase: 1})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ... {:.2f}s passed".format(i, time.time()-starttime))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
        
    saver.save(sess, './lenet')
    print("Model saved")

# %%

#Test set
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))