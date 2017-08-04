import os
import sys
import subprocess
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings
import util
import embed_n2v as n2v

# pipeline is
# Given a list of node files,
# embed each of the graphs with method X
# save the embeddings to an output directory
# return the embeddings to the user
# user evaluates those embeddings
# saves evaluated embedding results to a different folder

def embed(input_directory,
          output_directory,
          method = 'n2v',
          sample = -1):

    if method == 'n2v': 
        return n2v.embed_n2v(input_directory, output_directory,
                        weighted = False, verbose = False,
                        sample = sample, selected_nodes = "")

    if method == 'gcn':
        return -1 # not implemented yet, need to include training phase as well

embeddings = embed('/dfs/scratch0/manans/small-ppi-test',
                    '/dfs/scratch0/manans/small-ppi-test-output', 
                    method='n2v', sample = 100)

print embeddings
