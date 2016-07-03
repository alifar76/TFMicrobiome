import tensorflow as tf
import numpy as np
import input_read

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ksize: A list of ints that has length >= 4. 
# ksize is the size of the window for each dimension of the input tensor.
# stride: A list of ints that has length >= 4. 
# stride is the stride of the sliding window for each dimension of the input tensor.
# 3rd element in the list indicates the number of features/OTUs in dataset
# Changing ksize feature value to 2 doesn't impact much. 
# Changing to 15 also has little impact.

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 1, 1],
                        strides=[1, 1, 1, 1], padding='SAME')


## Input file specific variables
otuinfile = 'lozupone_hiv.txt'
metadata = 'mapfile_lozupone.txt'
# Split 55% of data as training and 45% as test
train_ratio = 0.55
metavar = ['hiv_stat','HIV status']  
levels = ['HIV_postive','HIV_negative','Undetermined']
# Read data
data = input_read.data_read(otuinfile,metadata,train_ratio,metavar,levels)
train_dataset = data[0]
test_input = data[1]
test_output = data[2]


# Data specific
feature = 267
resp = 2
# These variables are not having impact on accuracy
first_conv_feature = 32
second_conv_feature = 64
dense_layer_feature = 1024
opt_param = 1e-1



x = tf.placeholder(tf.float32, [None, feature])
y_ = tf.placeholder(tf.float32, [None, resp])
# First Convolutional Layer (changed filter value to 1x267)
W_conv1 = weight_variable([1, feature, 1, first_conv_feature])
b_conv1 = bias_variable([first_conv_feature])
# Reshape 267 features into 1*267 matrix
x_image = tf.reshape(x, [-1,1,feature,1])
#x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)


# Second Convolutional Layer (changed filter to 1x267)
# If ksize is 2, then feature value is 134 for 2nd conv layer
W_conv2 = weight_variable([1, feature, first_conv_feature, second_conv_feature])
b_conv2 = bias_variable([second_conv_feature])
# Not doing pooling. 
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  #h_pool1
#h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
# Values of 23 * 1 * 64 come from printing (h_pool2)
# If ksize is 2, then feature value is 67 for densly connected layer
W_fc1 = weight_variable([1 * feature * second_conv_feature, dense_layer_feature])
b_fc1 = bias_variable([dense_layer_feature])
# Not doing pooling
h_pool2_flat = tf.reshape(h_conv2, [-1, 1*feature*second_conv_feature]) #h_pool2
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# Dropout or no drop-out has no impact
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([dense_layer_feature, resp])
b_fc2 = bias_variable([resp])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #h_fc1_drop

# Train and evaluate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))



train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.95, epsilon=1e-02).minimize(cross_entropy)    #AdamOptimizer(opt_param)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for index, row in train_dataset.iterrows():
  # Store 0th-index is HIV postive and 1st-index is HIV negative
  store = [0,0]
  otudat = row.drop('HIV status', axis=0).values
  if row['HIV status'] == 'HIV_postive':
    store[0] = 1
  else:
    store[1] = 1
  response = np.asarray(store)
  batch_xs = np.reshape(otudat, (-1, 267))
  batch_ys = np.reshape(response, (-1, 2))
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: np.asarray(test_input), y_: np.asarray(test_output),keep_prob: 1.0}))
