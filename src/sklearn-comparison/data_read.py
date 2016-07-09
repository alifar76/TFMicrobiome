import numpy as np
import pandas as pd
import math


def load():
    ## Input file specific variables
    otuinfile = '../lozupone_hiv.txt'
    metadata = '../mapfile_lozupone.txt'
    # Split 55% of data as training and 45% as test
    train_ratio = 0.55
    metavar = ['hiv_stat','HIV status']
    levels = ['HIV_postive','HIV_negative','Undetermined']
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
    n_train = int(math.ceil(train_ratio*c.shape[0]))
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    train_dataset = c[:n_train]
    test_dataset = c[n_train:]
    # Order: X, Y, P, Q
    return [train_dataset.drop(metavar[1],1),
    train_dataset[[metavar[1]]],
    test_dataset.drop(metavar[1],1),
    test_dataset[[metavar[1]]]]