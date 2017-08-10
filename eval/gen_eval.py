# general evaluation script 
# uses a graph type and increases complexity
# checks if two graphs are closer than a third via embeddings

import snap
import numpy as np
from scipy.spatial import distance
import random
from tqdm import tqdm
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
            G = snap.GenStar(snap.PUNGraph, self.nnodes, False)
        elif self.graph_type == 'full':
            G = snap.GenFull(snap.PUNGraph, self.nnodes)
        elif self.graph_type == 'circle':
            G = snap.GenCircle(snap.PUNGraph, self.nnodes, False)
        elif self.graph_type == 'tree':
            G = snap.GenTree(snap.PUNGraph, Fanout=2, Levels=10, IsDir=False)
        elif self.graph_type == 'erdos-renyi' or self.graph_type == 'er':
            G = snap.GenRndGnm(snap.PUNGraph, self.nnodes, self.nnodes * 5, False)
        elif self.graph_type == 'barabasi-albert' or self.graph_type == 'ba':
            G = snap.GenPrefAttach(self.nnodes, 10)
        else:
            print "> Defaulting to full mode"
            G = snap.GenFull(snap.PUNGraph, self.nnodes)

        return G

    def graph(self):
        return self.graph_list[0]

    def generate_graphs(self, base_graph, k):
        # generates k permutations of increasing complexity 
        n_edges = base_graph.GetEdges()
        graphs = [base_graph]
        for i in range(1, k+1):
            graphs.append(snap.GenRewire(base_graph, int((float(i) / (k + 1)) * n_edges)) )
        self.graph_list = graphs

    def initialize(self):
        base = self.base_graph()
        self.generate_graphs(base, self.k)
        for i, g in enumerate(self.graph_list):
            snap.SaveEdgeList(g, self.graph_directory + '/' + str(i) + '.edgelist')
        
        return self.graph_directory

    def matrix(self, embeddings, metric=distance.sqeuclidean):
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

    def sample(self, iterator, k):
        # fill the reservoir to start
        result = [next(iterator) for _ in range(k)]

        n = k - 1
        for item in iterator:
            n += 1
            s = random.randint(0, n)
            if s < k:
                result[s] = item

        return result

    def evaluate(self, embeddings, N = 150, metric=distance.cosine, verbose=True):
        m = self.matrix(embeddings, metric)
        graph_ids = [i for i in xrange(len(embeddings))]
        y = 0
        n = 0
        for i in tqdm(xrange(N), disable = not verbose):
            ids = self.sample(iter(graph_ids), 3)
            while(abs(ids[1] - ids[0]) == abs(ids[2] - ids[1])):
                ids = self.sample(iter(graph_ids), 3)
            
            d_0_1 = m[ids[0]][ids[1]]
            d_1_2 = m[ids[1]][ids[2]]
           
            if abs(ids[1] - ids[0]) > abs(ids[2] - ids[1]): # 0 and 1 are further than 1 and 2
                if d_0_1 > d_1_2:
                    y += 1
                else:
                    n += 1
            else:
                if d_0_1 < d_1_2:
                    y += 1
                else:
                    n += 1
        return y, n

    def evaluate_random(self, N = 150, metric=distance.cosine, verbose=True):
        graph_ids = [i for i in xrange(len(self.graph_list))]
        embeddings = {str(graph_ids[i]) : np.random.randn(128) for i in graph_ids}
        return self.evaluate(embeddings, N, metric, verbose)
