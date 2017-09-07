# tree_eval evaluates embeddings by comparing their similarity with tree distances
import os
import sys
import subprocess
from tqdm import tqdm

sys.path.insert(0, '../common')
import settings
import util

from scipy.spatial import distance
from ete2 import Tree
import numpy as np

import random
from operator import itemgetter

import warnings
from random import shuffle

class TreeEvaluator():
    def __init__(self, vectors, path=settings.SPECIES_TREE, baseline=False):
        self.path = path
        self.tree = Tree(path)
        self.seed = 0
        
        random.seed(self.seed)
       
        if baseline:
            # if baseline is True, randomly shuffle the vectors
            values = vectors.values()
            shuffle(values)
            vectors = dict(zip(vectors.keys(), values)) 

        self.vector_map = vectors
 
        # update the tree with the vectors -- only need to do this once
        for leaf in self.tree.get_leaves():
            try:
                leaf.add_features(vector=vectors[leaf.name])
            except Exception as e:
                leaf.add_features(vector=[1]*len(vectors.values()[0]))
        
    def sample(self, l, k):
        return [l[i] for i in sorted(random.sample(xrange(len(l)), k))]

    def distances(self, nodes):
        d = []
        for i in range(0, len(nodes)):
            for j in range(i, len(nodes)):
                if j != i:
                    d.append(nodes[i].get_distance(nodes[j]))
        return d

    def get_nodes(self, min_dist, max_dist, k=3):

        all_nodes = self.tree.get_leaves()
        nodes = self.sample(all_nodes, k)
        distances = self.distances(nodes)

        while(False in [(i > min_dist and i < max_dist) for i in distances]):
            nodes = self.sample(all_nodes, k)
            distances = self.distances(nodes)

        return nodes, distances

    def evaluate(self,
                 n_perm = 100,      # number of times test will be executed
                 min_dist = 3,      # minimum distance between two nodes -- smaller makes the task harder
                 max_dist = 1000,   # maximum distance between two nodes -- smaller makes the task harder
                 metric = distance.cosine): # vector similarity metric

        n_correct = 0       # tree distance and vector distance both greater
        n_incorrect = 0     # tree distance and vector distance of opposite signs
        avg_corr = 0.       # average correlation coefficient

        progress = tqdm(total=n_perm)

        for p in xrange(n_perm):
            nodes, distances = self.get_nodes(min_dist, max_dist, k=3)
            distances_tst = []
            progress.set_description(str(distances))
            for i in range(0, len(nodes)):
                for j in range(i, len(nodes)):
                    if j != i:
                        distances_tst.append(metric(nodes[i].vector, nodes[j].vector))

            diff1 = nodes[1].get_distance(nodes[0]) - nodes[1].get_distance(nodes[2])
            diff2 = metric(nodes[1].vector, nodes[0].vector) - metric(nodes[1].vector, nodes[2].vector)

            if diff1 > 0 and diff2 > 0 or diff1 < 0 and diff2 < 0:
                n_correct += 1
            else:
                n_incorrect += 1

            cc = np.corrcoef(distances, distances_tst)[0][1]
            if np.isnan(cc):
                cc = 0.
            
            progress.update(1)

            avg_corr += np.abs(cc)
        avg_corr /= n_perm
        progress.close()

        return avg_corr, n_correct, n_incorrect 
