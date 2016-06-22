import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import math


otuinfile = 'lozupone_hiv.txt'
metadata = 'mapfile_lozupone.txt'
train_ratio = 0.66


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
n_train = math.ceil(train_ratio*c.shape[0])
n_test = c.shape[0] - n_train

for index, row in c.iterrows():
  print row.drop('HIV status', axis=0).values
  #print row




"""filename_queue = tf.train.string_input_producer(["test.csv"])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Num of features
num_feature = 4
## The values of 1 indicate int32
record_defaults = [['label']]+[[1]]*num_feature
data = tf.decode_csv(value,record_defaults)
features = tf.pack(data[1:])
with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, data[0]])
    #print example, label
  coord.request_stop()
  coord.join(threads)


infile = "test.csv"
with open(infile) as f:
    ncols = len(f.readline().split(','))
a = np.loadtxt(infile,delimiter=",",skiprows=1,usecols=range(1,ncols+1))"""








"""samples = []
infile = open(otuinfile,'rU')
for line in infile:
  if line.startswith("# Constructed"):
    pass
  if line.startswith("#OTU"):
    spline = line.strip().split("\t")
    samples = spline[1:]

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

u = [x for x in list(response.keys()) if x not in samples]


samples = []
infile = open(otuinfile,'rU')
for line in infile:
  if not line.startswith("#"):
    spline = line.strip().split("\t")
    print (spline)"""




#SampleID, hiv_stat, status





"""mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print (len(mnist.train.images))

print(len(mnist.test.labels))
print (len(mnist.test.images))
#print (len(mnist.validation.images))


print (len(mnist.test.images[0]))
print (len(mnist.test.labels[0]))



#features = tf.pack(data)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))"""


"""# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.pack([col1, col2, col3, col4])


with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])
    print example, label

  coord.request_stop()
  coord.join(threads)"""