# TFMicrobiome

A proof-of-concept demo using deep-learning for diagnostics

Background
------

This is a proof-of-concept pipeline based on the [TensorFlow](https://www.tensorflow.org/) library developed by Google. The idea is simple: using a publicly available microbiome dataset, we wish to develop a deep-learning based diagnostic platform using TensorFlow.
 
For this purpose, dataset corresponding to a [microbiome study by Lozupone et al.](http://www.ncbi.nlm.nih.gov/pubmed/24034618), was obtained from the [Qiita database](https://qiita.ucsd.edu/). 

A microbiome dataset consists of a count matrix in which the bacteria characterized by [Next Generation Sequencing](http://www.illumina.com/technology/next-generation-sequencing.html) are the rows and the samples from which the bacterial count information is obtained are columns. Additionally, there is meta-data associated with the samples. On a technical note, the bacterial count here refers to [OTUs](http://www.drive5.com/usearch/manual/otu_definition.html).
 
In this POC analysis, we are using the infection status of the individuals (HIV positive vs. negative) as the response variable and count of bacteria (OTUs) as explanatory variables in the model building process. For simple illustrative purposes, we will assess the accuracy
 
Currently, work in under progress for this pipeline.  The README file along with the otu_tensor.py script will keep on changing accordingly. 
