# general evaluation function for embeddings
import os
import sys
import subprocess
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/eval/')
import settings
import util
from embed import embed
from gen_eval import GraphEvaluator
import numpy as np

def accuracy(tup):
    return tup[0] / float(tup[0] + tup[1])

def run_x(fun):
    vals = [accuracy(fun) for i in xrange(100)]
    print vals

print "#1 -> Creating Graphs"
e = GraphEvaluator(graph_type = 'star', nnodes = 500, k = 10, graph_directory = '/dfs/scratch0/manans/rewire-exp')
e.initialize(rewire_edge_percent = True)

print "#2 -> Running Evaluation"

V_R = [accuracy(e.evaluate_random(verbose = False)) for i in xrange(100)]
print "R =>", np.mean(V_R), np.median(V_R), np.std(V_R)

''' Evaluation 1: Graph Type v Post-Rewiring Accuracy'''
for s in [5, 10, 50, 100, 250, -1]:

    embeddings = embed('/dfs/scratch0/manans/rewire-exp', '/dfs/scratch0/manans/rewire-exp-emb',
                    method = 'n2v', sample = s, verbose = False)

    result = e.evaluate(embeddings, verbose = False)
    V = [accuracy(e.evaluate(embeddings, verbose = False)) for i in xrange(100)]
    print str(s) + " =>", np.mean(V), np.median(V), np.std(V)
