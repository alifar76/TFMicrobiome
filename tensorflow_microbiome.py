import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import math


otuinfile = 'lozupone_hiv.txt'
metadata = 'mapfile_lozupone.txt'
# Split 55% of data as training and 45% as test
train_ratio = 0.55


a = pd.read_table(otuinfile,skiprows=1,index_col=0)
b = a.transpose()
response = {}
hiv = 0
infile = open(metadata,'rU')
for line in infile:
  if line.startswith("#SampleID"):
    spline = line.strip().split("\t")
    hiv = spline.index('hiv_stat')
  else:
    spline = line.strip().split("\t")
    response[spline[0]] = spline[hiv]
u = [response[x] for x in list(b.index)]
v = ['HIV_postive' if x == 'True' else 'HIV_negative' if x == 'False' else 'Undetermined' for x in u]
b.loc[:,'HIV status'] = pd.Series(v, index=b.index)
c = b[b['HIV status'].isin(['HIV_postive', 'HIV_negative'])]
# No. of samples to train/test the model
n_train = int(math.ceil(train_ratio*c.shape[0]))

train_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()
train_dataset = c[:n_train]
test_dataset = c[n_train:]


test_input = []
test_output = []
for index, row in test_dataset.iterrows():
  # Store 0th-index is HIV postive and 1st-index is HIV negative
  store = [0,0]
  otudat = row.drop('HIV status', axis=0).values
  if row['HIV status'] == 'HIV_postive':
    store[0] = 1
  else:
    store[1] = 1
  test_input.append(otudat)
  test_output.append(store)

#print train_dataset.shape, test_dataset.shape



###### TF code
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