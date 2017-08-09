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

def accuracy(tup):
    return tup[0] / float(tup[0] + tup[1])

print "#1 -> Creating Graphs"
e = GraphEvaluator(graph_type = 'erdos-renyi', nnodes = 500, k = 10, graph_directory = '/dfs/scratch0/manans/rewire-exp')
e.initialize()

print "#2 -> Running Evaluation"
for s in [100, 50, 10]:

    embeddings = embed('/dfs/scratch0/manans/rewire-exp', '/dfs/scratch0/manans/rewire-exp-emb',
                    method = 'n2v', sample = s, verbose = False)

    result = e.evaluate(embeddings, verbose = False)
    print embeddings['0']
    print s, result, accuracy(result)

