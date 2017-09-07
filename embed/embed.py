import os
import sys
import subprocess
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/embed-nf/original/')
import settings
import util
import embed_n2v as n2v
import embed_nf_original as nfo

def embed(train_input_directory,    # directory of train graphs
          train_label_mapping,      # map from train graph name -> label
          test_input_directory,     # directory of test graphs
          test_output_directory,    # directory to output test graph embeddings
          method,                   # embedding method

          # node2vec
          sample = -1,
          selected_nodes = "",

          # nf-original
          tmp_dir = "/dfs/scratch0/manans/tmp.csv",
          weights = "/dfs/scratch0/manans/results.pkl",
          train = True,
          n_epochs = 15,

          verbose = True,
          overwrite = True):

    if method == 'n2v': 
        return n2v.embed_n2v(test_input_directory, test_output_directory,
                        weighted = False, verbose = verbose,
                        sample = sample, selected_nodes = selected_nodes,
                        overwrite = overwrite)

    if method == 'gcn':
        return -1 # not implemented yet, need to include training phase as well

    if method == 'nf-o' or method == 'nf-original':
        if train:
            nfo.train_nf_original(train_input_directory, train_label_mapping,
                                tmp_dir = tmp_dir, output_directory = weights,
                                n_epochs = n_epochs)

        return nfo.embed_nf_original(test_input_directory, test_output_directory,
                                    weights = weights,
                                    verbose = verbose, overwrite = overwrite)

    if method == 'nf-k' or method == 'nf-keras':
        return -1 # need to implement this 
