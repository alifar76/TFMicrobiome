import tensorflow as tf
from numpy import asarray, reshape
import input_read

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride_feat):
  return tf.nn.conv2d(x, W, strides=[1, 1, stride_feat, 1], padding='SAME')

# ksize: A list of ints that has length >= 4. 
# ksize is the size of the window for each dimension of the input tensor.
# stride: A list of ints that has length >= 4. 
# stride is the stride of the sliding window for each dimension of the input tensor.
# 3rd element in the list of ksize indicates the number of features/OTUs in dataset
# Changing ksize feature value to 2 doesn't impact much. 
# Changing to 15 also has little impact.

def max_pool_n(x,maxp_k,maxp_str):
  return tf.nn.max_pool(x, ksize=[1, 1, maxp_k, 1],
                        strides=[1, 1, maxp_str, 1], padding='SAME')


def conv_layer(input_feat,conv_feature,x_shape,conv2d_stride,
  maxpool_ksize,maxpool_stride,in_ch):
  """ Convolutional layer """
  W_conv1 = weight_variable([1, input_feat, in_ch, conv_feature])
  b_conv1 = bias_variable([conv_feature])
  h_conv1 = tf.nn.relu(conv2d(x_shape, W_conv1,conv2d_stride) + b_conv1)
  #red_feat_dim1 = h_conv1.get_shape().as_list()[2]
  h_pool1 = max_pool_n(h_conv1,maxpool_ksize,maxpool_stride)
  # In order to get the reduced dimension of feature vector, we do following:
  # 2 is the index of the 4d tensor
  red_feat_dim1 = h_pool1.get_shape().as_list()[2]
  if (red_feat_dim1 < conv2d_stride):
    conv2d_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_stride):
    maxpool_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_ksize):
    maxpool_ksize = red_feat_dim1
  return [h_pool1,red_feat_dim1,conv2d_stride,maxpool_ksize,maxpool_stride,conv_feature]


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

# Data specific:
# No. of features
feature = 267
# No. of classification categories
resp = 2

# These variables are not having impact on accuracy
first_conv_feature = 32
second_conv_feature = 64
dense_layer_feature = 1024
opt_param = 1e-1
# For AdadeltaOptimizer()
a = 0.1  # learning_rate
b = 0.95  # rho
c = 1e-02  # epsilon


# Value to stride for conv2d
conv2d_stride = 1
# Value to stride and ksize for maxpool
maxpool_ksize = 1
maxpool_stride = 1
str_siz = [conv2d_stride,maxpool_ksize,maxpool_stride]

x = tf.placeholder(tf.float32, [None, feature])
y_ = tf.placeholder(tf.float32, [None, resp])
# Reshape 267 features into 1*267 matrix
x_shape = tf.reshape(x, [-1,1,feature,1])


# Call convolutional layers
# The last argument, 1, is the value of first convolution input channel
conv1 = conv_layer(feature,first_conv_feature,x_shape,
  str_siz[0],str_siz[1],str_siz[2],1)
conv2 = conv_layer(conv1[1],second_conv_feature,conv1[0],
  conv1[2],conv1[3],conv1[4],conv1[5])


# Densely Connected Layer
# Values of 23 * 1 * 64 come from printing (h_pool2)
# If ksize is 2, then feature value is 67 for densly connected layer
W_fc1 = weight_variable([1 * conv2[1] * second_conv_feature, dense_layer_feature])
b_fc1 = bias_variable([dense_layer_feature])
# Not doing pooling
h_pool2_flat = tf.reshape(conv2[0], [-1, 1*conv2[1]*second_conv_feature]) #h_conv2
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# Dropout or no drop-out has no impact
k_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, k_prob)

# Readout Layer
W_fc2 = weight_variable([dense_layer_feature, resp])
b_fc2 = bias_variable([resp])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #h_fc1_drop

# Train and evaluate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))


#AdamOptimizer(opt_param)
train_step = tf.train.AdadeltaOptimizer(a,b,c).minimize(cross_entropy)
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
  response = asarray(store)
  batch_xs = reshape(otudat, (-1, 267))
  batch_ys = reshape(response, (-1, 2))
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, k_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: asarray(test_input), y_: asarray(test_output),k_prob: 1.0}))