import os
import sys
from tqdm import tqdm
import numpy as np
from time import time

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/eval/')
import settings
import util
from tree_eval import TreeEvaluator

from os import listdir
from os.path import isfile, join

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

from scipy.spatial import distance

EMBEDDING_FILE = 'emb/n2v-avg.nemb' 
vector_map = {}

with open(EMBEDDING_FILE, 'r') as f:
    for line in f:
        line = line.split(' ')
        n = line[0]
        v = []
        for val in line[1:]:
            v.append(float(val))
        
        vector_map[n] = v

evaluator = TreeEvaluator(vectors=vector_map, baseline=False)

for i in [10]: #[0, 5, 10, 15, 20]:
    print "[new] MIN_DIST = " + str(i)
    corr, y, n = evaluator.evaluate(n_perm = 5000, min_dist = i, metric=distance.sqeuclidean)
    print "\t", corr, y, n

evaluator = TreeEvaluator(vectors=vector_map, baseline=True)

for i in [10]: #[0, 5, 10, 15, 20]:
    print "[baseline] MIN_DIST = " + str(i)
    corr, y, n = evaluator.evaluate(n_perm = 5000, min_dist = i, metric=distance.sqeuclidean)
    print "\t", corr, y, n
