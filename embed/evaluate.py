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
sys.path.insert(0, '../embed-n2v')
sys.path.insert(0, '../common')
sys.path.insert(0, '../eval')

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

def stats(l):
    return str(np.mean(l)), str(np.std(l))

def run(graph_type, nnodes, k, f):
    base_dir = '/dfs/scratch0/manans/rewire-exp-' + graph_type
    
    # clean directories
    os.system('rm ' + base_dir + '/train/*')
    os.system('rm ' + base_dir + '/test/*')
    os.system('rm ' + base_dir + '-emb/train/*')
    os.system('rm ' + base_dir + '-emb/test/*')

    e = GraphEvaluator(graph_type = graph_type,
                        nnodes = nnodes, k = k, 
                        graph_directory = base_dir)
    e.initialize()

    header = graph_type + \
            '_k=' + str(k) + \
            '_n=' + str(nnodes) + \
            '_e=' + str(e.graph().GetEdges()) + \
            '_t=' + datetime.datetime.now().isoformat().split('.')[0] + ','

    means = " ,"
    stds  = " ,"

    '''
    # method 1: random
    header += "random,"
    v = stats([accuracy(e.evaluate_random()) for i in xrange(100)])
    means += v[0] + "," 
    stds  += v[1] + ","

    # method 2-11: node2vec
    for i in tqdm([10, 20, 30, 40, 50, 60, 70, 80, 90, -1]):
        header += ("n2v-0." + str(i) + "," if i > 0 else "n2v-1.0,")

        embeddings = embed(train_input_directory    = None,
                            train_label_mapping     = None,
                            test_input_directory    = base_dir + '/test',
                            test_output_directory   = base_dir + '-emb/test',
                            method = 'n2v', 
                            sample = int((float(i) / 100) * nnodes) if i > 0 else -1, 
                            verbose = False)

        v = stats([accuracy(e.evaluate(embeddings)) for i in xrange(100)])
        means += v[0] + "," 
        stds  += v[1] + ","

        os.system('rm ' + base_dir + '-emb/train/*')
        os.system('rm ' + base_dir + '-emb/test/*')
    '''
    # method 12: nf-original
    header += "nf-original,"
    train_labels = {}
    for fl in os.listdir(base_dir + '/train'):
        train_labels[fl] = float(fl.split('.')[0])/k

    embeddings = embed(train_input_directory    = base_dir + '/train',
                        train_label_mapping     = train_labels,
                        test_input_directory    = base_dir + '/test', 
                        test_output_directory   = base_dir + '-emb/test/',
                        method = 'nf-original', 
                        train = True,
                        verbose = False)

    v = stats([accuracy(e.evaluate(embeddings)) for i in xrange(100)])
    means += v[0] + ","
    stds  += v[1] + ","

    # remove ending comma
    header = header[:-1]
    means = means[:-1]
    stds = stds[:-1]

    f.write(header + "\n")
    f.write(means + "\n")
    f.write(stds + "\n")

    print(header)
    print(means)
    print(stds)
    print(">>>------")

for i in range(100, 500, 100): # i is nnodes
    run('er', i, 2000, out)
    sys.exit(0)
    run('ba', i, 2000, out)

out.close()
