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