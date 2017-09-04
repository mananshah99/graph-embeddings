'''
august 10 evaluation table generation code
first column is graph name _ number of nodes _ number of edges
then k columns, where each column reports accuracy when a different method is used 
    node2vec_sampling_0.05
        etc. 

        K = 50 (but still a fraction)
'''

import os
import sys
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/embed-n2v')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/eval/')

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

EXP_FILE = "exp_begin-aug10.csv"
out = open(EXP_FILE, "a")

def accuracy(tup):
    return tup[0] / float(tup[0] + tup[1])

def metrics(l):
    return str(np.mean(l)), str(np.std(l))

def run(graph_type, nnodes, f):
    os.system('rm /dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb/*')

    # every instance of 'run' operates on the same set of graphs, initialized
    # by instance 'e' of GraphEvaluator
    e = GraphEvaluator(graph_type = graph_type,
                        nnodes = nnodes, 
                        k = 100, 
                        graph_directory = '/dfs/scratch0/manans/rewire-exp-' + graph_type)
    e.initialize()

    header = graph_type + '_k=' + str(50) + '_n=' + str(nnodes) + '_e=' + str(e.graph().GetEdges()) + '_t=' + datetime.datetime.now().isoformat().split('.')[0] + ','
    means = " ," # first column has no associated values, just graph description
    stds  = " ,"

    '''
    # method 1: random
    header += "random,"
    v = metrics([accuracy(e.evaluate_random(verbose = False)) for i in xrange(100)])
    means += v[0] + "," 
    stds  += v[1] + ","

    # method 2-11: node2vec [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in tqdm([10, 20, 30, 40, 50, 60, 70, 80, 90, -1]):
        header += ("n2v-0." + str(i) + "," if i > 0 else "n2v-1.0,")

        # only embed the test set (discard train)
        embeddings = embed('/dfs/scratch0/manans/rewire-exp-' + graph_type + '/test',
                            '/dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb/test',
                            method = 'n2v', sample = int((float(i) / 100) * nnodes) if i > 0 else -1, verbose = False)

        v = metrics([accuracy(e.evaluate(embeddings, verbose = False)) for i in xrange(100)])
        means += v[0] + "," 
        stds  += v[1] + ","

        os.system('rm /dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb/train/*')
        os.system('rm /dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb/test/*')

    '''
    
    # method 12: nf-original

    embeddings = embed('/dfs/scratch0/manans/rewire-exp-' + graph_type + '/train/',
                        '/dfs/scratch0/manans/rewire-exp-' + graph_type + '/test/', 
                        '/dfs/scratch0/manans/rewire-exp-' + graph_type + '-emb/test/',
                        method = 'nf-original')


    #TODO: Write method 13: hamilton graphSAGE with train and test directories


    # remove ending comma
    header = header[:-1]
    means = means[:-1]
    stds = stds[:-1]

    f.write(header + "\n")
    f.write(means + "\n")
    f.write(stds + "\n")

for i in range(50, 500, 50): # i is nnodes
    run('er', i, out)
    run('ba', i, out)

out.close()
