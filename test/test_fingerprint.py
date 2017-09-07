import os, sys
sys.path.append("../embed")

from embed import embed

# Directories with stored graph edgelist representations
train_graph_directory     = 'train_graphs/'
train_graph_labels        = {x : int(x.split('.')[0]) % 2 for x in os.listdir(train_graph_directory)}
test_graph_directory      = 'test_graphs/'
test_graph_output         = 'test_graph_embeddings/'

embeddings = \
    embed(train_input_directory     = train_graph_directory, # nf is supervised
            train_label_mapping     = train_graph_labels,
            test_input_directory    = test_graph_directory,
            test_output_directory   = test_graph_output,
            method                  = 'nf-o',
            tmp_dir                 = 'bin/tmp.csv',
            weights                 = 'bin/weights.pkl', 
            train                   = True,
            n_epochs                = 1)

