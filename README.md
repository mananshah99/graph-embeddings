# graph-embeddings

Code Listing

| Name                   | Description                                                                  |
|------------------------|------------------------------------------------------------------------------|
| ``common/util.py``     | General utility functions to simplify file i/o                               |
| ``common/log.py``      | Functions to aid in logging program output                                   |
| ``common/settings.py`` | Static variables for file and directory locations                            |
| ``split_full.py``      | Transforms ``protein.links.full.v10.txt`` to 2,000 individual PPI networks   |
| ``create_graph.py``    | Transforms 2,000 individual PPI networks into edgelist representations       |
| ``embed_node2vec.py``  | Embeds edgelist representations with node2vec uniform random sampling        |
| ``process_node2vec.py``| Processes all node2vec embeddings to obtain a standardized histogram / graph | 
| ``cog_tests/cog.py``   | Run all COG experiments (inter/intra species + table visualizations)         |

Relevant Directory Listing

| Name                                 | Description                                                                |
|--------------------------------------|----------------------------------------------------------------------------|
| ``/dfs/scratch0/manans/ppi``         | Raw representations of all PPI networks (including all relevant columns)   | 
| ``/dfs/scratch0/manans/ppi-uw-graph``| Unweighted PPI graphs (binary indications for edges)                       |
| ``/dfs/scratch0/manans/ppi-w-graph`` | Weighted PPI graphs (edge weights calculated as sum of columns)            |
| ``/dfs/scratch0/manans/ppi-uw-n2v``  | Node2vec embeddings of unweighted PPI graphs (dim:128, s:100)              | 
