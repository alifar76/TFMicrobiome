import numpy as np
import pandas as pd
import math


def data_read(otuinfile,metadata,train_ratio,metavar,levels):
  """ Reads OTU table data and meta data and creates train/test dataset """
  a = pd.read_table(otuinfile,skiprows=1,index_col=0)
  b = a.transpose()
  response = {}
  hiv = 0
  infile = open(metadata,'rU')
  for line in infile:
    if line.startswith("#SampleID"):
      spline = line.strip().split("\t")
      hiv = spline.index(metavar[0])
    else:
      spline = line.strip().split("\t")
      response[spline[0]] = spline[hiv]
  u = [response[x] for x in list(b.index)]
  v = [levels[0] if x == 'True' else levels[1] if x == 'False' else levels[2] for x in u]
  b.loc[:,metavar[1]] = pd.Series(v, index=b.index)
  c = b[b[metavar[1]].isin([levels[0], levels[1]])]
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
    otudat = row.drop(metavar[1], axis=0).values
    if row[metavar[1]] == levels[0]:
      store[0] = 1
    else:
      store[1] = 1
    test_input.append(otudat)
    test_output.append(store)
  return [train_dataset, test_input, test_output]
