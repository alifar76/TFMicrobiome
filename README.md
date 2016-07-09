# TFMicrobiome

A proof-of-concept demo of deep-learning for diagnostics (using microbiome data).

DISCLAIMER
------

Currently, work is in under progress for this proof-of-concept project. The README file along with the Python scripts will keep on changing accordingly. 


Background
------

This is a proof-of-concept pipeline based on the [TensorFlow](https://github.com/tensorflow/tensorflow) library developed by Google. The idea is simple: using a publicly available microbiome dataset, we wish to develop a deep-learning, diagnostic platform based on TensorFlow.
 
For this purpose, dataset corresponding to a [microbiome study by Lozupone et al.](http://www.ncbi.nlm.nih.gov/pubmed/24034618), was obtained from the [Qiita database](https://qiita.ucsd.edu/). 

A microbiome dataset consists of a count matrix in which the bacteria characterized by [Next Generation Sequencing](http://www.illumina.com/technology/next-generation-sequencing.html) are the rows and the samples from which the bacterial count information is obtained are columns. Additionally, there is meta-data associated with the samples. On a technical note, the bacterial count here refers to [OTUs](http://www.drive5.com/usearch/manual/otu_definition.html).
 
In this POC analysis, we are using the infection status of the individuals (HIV positive vs. negative) as the response variable and count of bacteria (OTUs) as explanatory variables in the model building process. For simple illustrative purposes, we will assess the accuracy.

Required Packages
------

- [Enthought Canopy Python 2.7.6 | 64-bit](https://store.enthought.com/downloads/#default)
- [virtualenv 15.0.1](https://virtualenv.pypa.io/en/stable/installation/)
- [TensorFlow 0.9.0](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation)
- [NumPy 1.11.1](http://www.scipy.org/install.html)
- [pandas 0.18.1](https://pypi.python.org/pypi/pandas/0.18.1/#downloads)

The Mac OS X, CPU only, Python 2.7 version of TensorFlow was installed via pip in a virtualenv on my MacBook Pro having OS X El Capitan Version 10.11.4.

Additionally, I'm using [scikit-learn 0.17.1](http://scikit-learn.org/stable/install.html) to compare performance of more classical machine learning algorithms with deep learning methods in TensorFlow.

How to
------

There are two scripts, each of which implement a separate model. They can be simply run by the following command:

- ```python softmax.py```
- ```python cnn.py```

The output will the accuracy of the model.


Result
------

### TensorFlow models:
- ``` Accuracy of [Softmax Regression model](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html):  0.909```
- ``` Accuracy of [Convolutional Neural Network](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html): 0.773 ```

### scikit-learn models:
- ``` Accuracy of Random Forest Classifier: 0.864 ```
- ``` Accuracy of SVM: 0.864 ```
- ``` Accuracy of Gradient Boosting Classifier: 0.773 ```
- ``` Accuracy of Gaussian Naive Bayes Classifier: 0.909 ```
- ``` Accuracy of Multi-layer Perceptron Classifier: 0.864 ```
- ``` Accuracy of Blending: 0.909```

Other models will be implemented soon.

It's interesting to note that SVM, CNN, Gradient Boosting and SGD are giving almost identical performance. 
Gaussian NB and Softmax are also giving identical performance.
Experimentation with parameter tuning needs to be done more.
