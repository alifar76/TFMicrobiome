import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import input_read


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


###### TF for softmax regression
x = tf.placeholder(tf.float32, [None, 267])
W = tf.Variable(tf.zeros([267, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#GradientDescentOptimizer(0.5) can be used as well
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
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
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: np.asarray(test_input), y_: np.asarray(test_output)}))