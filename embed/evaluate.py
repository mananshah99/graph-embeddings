# general evaluation function for embeddings
import os
import sys
import subprocess
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/eval/')
import settings
import util
from embed import embed
from gen_eval import GraphEvaluator

e = GraphEvaluator(graph_type = 'full', nnodes = 10, k = 5, graph_directory = '/dfs/scratch0/manans/rewire-exp')
e.initialize()

embeddings = embed('/dfs/scratch0/manans/rewire-exp', '/dfs/scratch0/manans/rewire-exp-emb',
                    method = 'n2v', sample = -1)

print e.evaluate(embeddings)
