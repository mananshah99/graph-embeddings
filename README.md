# graph-embeddings

Methods to generate and evaluate *D*-dimensional feature vectors (embeddings) from unimodal, unweighted graphs of arbitrary complexity. A simple example is illustrated below: each call to the embed function returns a map of embeddings and saves the embeddings to a specified directory.

```py
import os
from embed import embed

# Directories with stored graph edgelist representations
train_graph_directory     = 'train_graphs/'
train_graph_labels        = {x : x.split('.')[0] for x in os.listdir(train_graph_directory)}
test_graph_directory      = 'test_graphs/'
test_graph_output         = 'test_graph_embeddings/'

node2vec_embeddings = \
	embed(train_input_directory 	        = None, # node2vec is unsupervised
		 	train_label_mappping 	= None,
		 	test_input_directory	= test_graph_directory,
		 	test_output_directory	= test_graph_output,
		 	method = 'n2v')

fingerprint_embeddings = \
	embed(train_input_directory 	        = train_graph_directory, # nf is supervised
		 	train_label_mappping 	= train_graph_labels,
		 	test_input_directory	= test_graph_directory,
		 	test_output_directory	= test_graph_output,
		 	method 			= 'nf-o',
			train			= True,
			n_epochs		= 15)

sage_embeddings = \
    	embed(train_input_directory     	= train_graph_directory, # sage is supervised
            		train_label_mapping     = train_graph_labels,
            		test_input_directory    = test_graph_directory,
            		test_output_directory   = test_graph_output,
            		method                  = 'sage',
			train 			= True,
            		n_epochs                = 15)
```

Numerous unique embedding methods were designed and implemented for this task; a brief description of their inner workings is as follows.

## Graph Convolutional Networks (embed-gcn)

Described by Thomas N. Kipf and Max Welling in "Semi-Supervised Classification with Graph Convolutional Networks" (arXiv, 2017). 

> [Graph Convolutional Networks (GCNs) are] a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. The choice of our convolutional architecture [is motivated] via a localized first-order approximation of spectral graph convolutions. [GCNs] scale linearly in the number of graph edges and learn hidden layer representations that encode both local graph structure and features of nodes.

Node feature vectors are learned in a supervised manner from a train set of data, and these node embeddings are aggregated and reduced to a fixed graph-wide embedding for the test phase.  

## Node2Vec (embed-n2v)

Described by Aditya Grover and Jure Leskovec in "node2vec: Scalable Feature Learning for Networks" (arXiv, 2016). 

> [node2vec is] an algorithmic framework for learning continuous feature representations for nodes in networks. [The algorithm] learns a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes.

The unsupervised nature of node2vec embeddings allows for the learning of node embeddings during the test phase directly; the node-level embeddings are aggregated to a graph-wide representation immediately afterward. 

## Neural Fingerprint (embed-nf)

Described by David Duvenaud et al. in "Convolutional Networks on Graphs for Learning Molecular Fingerprints" (arXiv, 2015). 

> [The neural fingerprint model] allow[s] end-to-end learning of prediction pipelines whose inputs are graphs of arbitrary size and shape. The architecture present[ed] generalizes standard molecular feature extraction methods based on circular fingerprints. 

Graph feature vectors are learned in a supervised manner, and the saved weights for the networks from the training phase are applied to generate graph embeddings in the test phase. 

## GraphSAGE (embed-sage)

Described by William L. Hamilton, Rex Ying, and Jure Leskovec in "Inductive Representation Learning on Large Graphs" (arXiv, 2017). 

> GraphSAGE [is] a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, [GraphSAGE] learn[s] a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. 

In a similar manner to the neural fingerprint method, graph vectors are learned in a supervised manner and supervised weights are applied to generate embeddings for unseen graphs in the test phase. 
