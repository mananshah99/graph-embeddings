'''
august 10 evaluation table generation code
first column is graph name _ number of nodes _ number of edges
then k columns, where each column reports accuracy when a different method is used 
    node2vec_sampling_0.05
        etc. 

        K = 50 (but still a fraction)
'''

import sys
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/eval/')

import subprocess
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import settings
import util
import numpy as np
from embed import embed
from gen_eval import GraphEvaluator
import datetime
from tqdm import tqdm

EXP_FILE = "exp_begin-aug10.csv"
out = open(EXP_FILE, "a")

def accuracy(tup):
    return tup[0] / float(tup[0] + tup[1])

def metrics(l):
    return str(np.mean(l)), str(np.std(l))

def run(graph_type, nnodes, f):
    e = GraphEvaluator(graph_type = graph_type, nnodes = nnodes, k = 50, graph_directory = '/dfs/scratch0/manans/rewire-exp-' + graph_type)
    e.initialize()
    header = graph_type + '_k=' + str(50) + '_n=' + str(nnodes) + '_e=' + str(e.graph().GetEdges()) + '_t=' + datetime.datetime.now().isoformat().split('.')[0] + ','
    
    means = " ," # first column has no associated values, just graph description
    stds  = " ,"

    # method 1: random
    header += "random,"
    v = metrics([accuracy(e.evaluate_random(verbose = False)) for i in xrange(100)])
    means += v[0] + "," 
    stds  += v[1] + ","

    # method 2-11: node2vec [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in tqdm([10, 20, 30, 40, 50, 60, 70, 80, 90, -1]):
        header += ("n2v-0." + str(i) + "," if i > 0 else "n2v-1.0,")
        embeddings = embed('/dfs/scratch0/manans/rewire-exp-' + graph_type,
                            '/dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb',
                            method = 'n2v', sample = int((float(i) / 100) * nnodes) if i > 0 else -1, verbose = False)

        v = metrics([accuracy(e.evaluate(embeddings, verbose = False)) for i in xrange(100)])
        means += v[0] + "," 
        stds  += v[1] + ","

    # remove ending comma
    header = header[:-1]
    means = means[:-1]
    stds = stds[:-1]

    f.write(header + "\n")
    f.write(means + "\n")
    f.write(stds + "\n")

run('er', 50, out)
run('ba', 50, out)

out.close()