# graph-embeddings

| Name                   | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| ``common/util.py``     | General utility functions to simplify file i/o
| ``common/log.py``      | Functions to aid in logging program output                                 |
| ``common/settings.py`` | Static variables for file and directory locations                          |
| ``split_full.py``      | Transforms ``protein.links.full.v10.txt`` to 2,000 individual PPI networks |
| ``create_graph.py``    | Transforms 2,000 individual PPI networks into edgelist representations     |
| ``embed_node2vec.py``  | Embeds edgelist representations with node2vec uniform random sampling      |
