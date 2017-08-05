# general evaluation script 
# uses a graph type and increases complexity
# checks if two graphs are closer than a third via embeddings

import snap
import numpy as np
from scipy.spatial import distance

class GraphEvaluator():
    def __init__(self, graph_type = 'erdos-renyi', nnodes = 1000,
                        k = 3, graph_directory = '.'):
        self.graph_type = graph_type
        self.graph_directory = graph_directory
        self.nnodes = nnodes
        self.graph_list = []
        self.k = k

    def base_graph(self):
        G = None
        if self.graph_type == 'star':
            G = snap.GenStar(snap.PUNGraph, self.nnodes, IsDir=False)
        elif self.graph_type == 'full':
            G = snap.GenFull(snap.PUNGraph, self.nnodes)
        elif self.graph_type == 'circle':
            G = snap.GenCircle(snap.PUNGraph, self.nnodes, IsDir=False)
        elif self.graph_type == 'tree':
            G = snap.GenTree(snap.PUNGraph, Fanout=2, Levels=10, IsDir=False)
        elif self.graph_tpye == 'erdos-renyi':
            G = snap.GenRndGnm(snap.PUNGraph, self.nnodes, self.nnodes * 5, IsDir=False)
        else:
            print "> Defaulting to full mode"
            G = snap.GenFull(snap.PUNGraph, self.nnodes)

        return G

    def generate_graphs(self, base_graph, k):
        # generates k permutations of increasing complexity 
        graphs = [base_graph]
        for i in range(1, k+1):
            graphs.append(snap.GenRewire(base_graph, i))
        self.graph_list = graphs

    def initialize(self):
        base = self.base_graph()
        self.generate_graphs(base, self.k)
        for i, g in enumerate(self.graph_list):
            snap.SaveEdgeList(g, self.graph_directory + '/' + str(i) + '.edgelist')
        
        return self.graph_directory

    def evaluate(self, embeddings, metric=distance.sqeuclidean):
        gl = self.graph_list
        assert len(gl) == len(embeddings)
        m = np.ndarray(shape=(len(gl), len(gl)))
        
        for i in xrange(len(gl)):
            for j in xrange(len(gl)):
                if i == j:
                    m[i][j] = 0
                else:
                    m[i][j] = metric(embeddings[str(i)], embeddings[str(j)])

        return m
