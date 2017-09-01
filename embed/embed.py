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

def embed(input_directory,
          output_directory,
          method = 'n2v',
          sample = -1,
          selected_nodes = "",
          verbose = True,
          overwrite = True):

    if method == 'n2v': 
        return n2v.embed_n2v(input_directory, output_directory,
                        weighted = False, verbose = verbose,
                        sample = sample, selected_nodes = selected_nodes,
                        overwrite = overwrite)

    if method == 'gcn':
        return -1 # not implemented yet, need to include training phase as well

    if method == 'nf-o' or method == 'nf-original':
        return nfo.embed_nf_original(input_directory, output_directory,
                                    werbose = verbose, overwrite = overwrite)

    if method == 'nf-k' or method == 'nf-keras':
        return -1 # need to implement this 
