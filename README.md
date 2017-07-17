# graph-embeddings

Code Listing

| Name                                 | Description                                                              |
|--------------------------------------|--------------------------------------------------------------------------|
| ``common/util.py``                   | General utility functions to simplify file i/o                           |
| ``common/log.py``                    | Functions to aid in logging program output                               |
| ``common/settings.py``               | Static variables for file and directory locations                        |
| ``process/split_full.py``            | Transforms ``protein.links.full.v10.txt`` to 2,031 individual networks   |
| ``process/create_graph.py``          | Transforms 2,031 individual PPI networks into edgelist representations   |
| ``embed-n2v/embed_n2v.py``           | Embeds edgelist representations with node2vec uniform random sampling    |
| ``embed-n2v/process_n2v.py``         | Processes all node2vec embeddings to obtain a standardized histogram     | 
| ``embed-n2v/visualize_tsne.py``      | Visualizes the processed embeddings in a low-dimensional space with tSNE |
| ``embed-n2v/evaluate_tree.py``       | Evaluates embedding effectiveness in comparison to the tree of life      |  
| ``embed-gcn/gen_subgraph.py``        | Generates subgraphs and labels from individual PPI networks              |
| ``embed-gcn/layers/graph.py``        | Defines the graph convolution layer in Keras form (from Kipf et al)      |
| ``embed-gcn/utils.py``               | Defines utilities for GCN data extraction and preprocessing              |
| ``embed-gcn/train_batch.py``         | Trains a GCN on a series of PPI networks                                 |
| ``eval/tree_eval.py``                | Framework to evaluate vectorized embeddings with permuted node  examples |
| ``exp-cog/cog.py``                   | Run all COG experiments (inter/intra species + table visualizations)     |

Relevant Directory Listing

| Name                                 | Description                                                                |
|--------------------------------------|----------------------------------------------------------------------------|
| ``/dfs/scratch0/manans/ppi``         | Raw representations of all PPI networks (including all relevant columns)   | 
| ``/dfs/scratch0/manans/ppi-uw-graph``| Unweighted PPI graphs (binary indications for edges)                       |
| ``/dfs/scratch0/manans/ppi-w-graph`` | Weighted PPI graphs (edge weights calculated as sum of columns)            |
| ``/dfs/scratch0/manans/ppi-uw-n2v``  | Node2vec embeddings of unweighted PPI graphs (dim:128, s:100)              | 
