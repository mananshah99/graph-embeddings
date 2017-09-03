from collections import OrderedDict, defaultdict
import numpy as np
import random
import dill as pickle


def _reverse_relation(relation):
    return (relation[-1], relation[1], relation[0])

class Graph():
    """
    Simple container for heteregeneous graph data.
    """
    def __init__(self, features, feature_dims, relations, adj_lists):
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.rel_edges = OrderedDict()
        self.edges = 0.
        self.nodes = {rel : np.array(adjs.keys()) for rel, adjs in adj_lists.iteritems()}
        self.degrees = {rel : np.array([len(adj_lists[rel][node]) 
            for node in self.nodes[rel]]) for rel in adj_lists}
        self.weights = {rel : degs/float(degs.sum()) for rel, degs in self.degrees.iteritems()}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1,r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in self.adj_lists[rel].values():
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += len(adj_list)

    def remove_edges(self, edge_list):
        for edge in edge_list:
            try:
                self.adj_lists[edge[-1]][edge[0]].remove(edge[1])
            except Exception:
                continue

            try:
                self.adj_lists[_reverse_relation(edge[-1])][edge[1]].remove(edge[0])
            except Exception:
                continue

    def add_edges(self, edge_list):
        for edge in edge_list:
            self.adj_lists[edge[-1]][edge[0]].append(edge[1])
            self.adj_lists[_reverse_relation(edge[-1])][edge[1]].append(edge[0])

    def get_all_edges(self, seed=0):
        """
        Returns all edges in the form (relation, node1, node2)
        """
        edges = []
        np.random.seed(seed)
        for rel, adjs in self.adj_lists.iteritems():
            for node, neighs in adjs.iteritems():
                edges.extend([(node, neigh, rel) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0):
        """
        Returns all edges in the form relation : list of edges
        """
        edges = defaultdict(list)
        np.random.seed(seed)
        for rel, adjs in self.adj_lists.iteritems():
            for node, neighs in adjs.iteritems():
                edges[rel].extend([(node, neigh, rel) for neigh in neighs if neigh != -1])
        for edge_list in edges.values():
            random.shuffle(edge_list)
        return edges

    def sample_relation(self):
        rel_index = np.argmax(np.random.multinomial(1, 
            np.array(self.rel_edges.values())/self.edges))
        rel = self.rel_edges.keys()[rel_index]
        return rel

    def sample_edge(self):
        """
        Samples an edge from the graph.
        Uniform across modes but biased towards low-degree.
        """
        rel_index = np.argmax(np.random.multinomial(1, 
            np.array(self.rel_edges.values())/self.edges))
        rel = self.rel_edges.keys()[rel_index]
        node = np.random.choice(self.nodes[rel], p=self.weights[rel])
        neigh = np.random.choice(self.adj_lists[rel][node])
        return node, neigh, rel

    def sample_negative_edge(self, rel=None):
        """
        Samples an edge from the graph.
        Uniform across modes but biased towards low-degree.
        """
        if rel is None:
            rel_index = np.argmax(np.random.multinomial(1, 
                np.array(self.rel_edges.values())/self.edges))
            rel = self.rel_edges.keys()[rel_index]
        node = np.random.choice(self.nodes[rel])
        neigh = np.random.choice(self.nodes[(rel[-1], rel[1], rel[0])])
        return node, neigh, rel

    
    def sample_path(self, length=2):
        """
        Samples a path from the graph.
        """
        rel_index = np.argmax(np.random.multinomial(1, 
            np.array(self.rel_edges.values())/self.edges))
        rel = self.rel_edges.keys()[rel_index]
        node = np.random.choice(self.nodes[rel], p=self.weights[rel])
        neigh = np.random.choice(self.adj_lists[rel][node])

        nodes = [neigh]
        rels = [rel]
        for i in range(length):
            rel_index = np.random.randint(len(self.relations[rels[-1][-1]]))
            next_rel = self.relations[rels[-1][-1]][rel_index]
            next_rel = (rels[-1][-1], next_rel[1], next_rel[0])
            if len(self.adj_lists[next_rel][node]) == 0:
                return self.sample_path(length=length)
            next_neigh = np.random.choice(
                    self.adj_lists[next_rel][node])
            nodes.append(next_neigh)
            rels.append(next_rel)

        return node, nodes[-1], rels 

def make_adj_lists(relations):
    adj_lists = {}
    for rel1, rel2s in relations.iteritems():
        for rel2 in rel2s:
            adj_lists[(rel1, rel2[1], rel2[0])] = defaultdict(list)
    return adj_lists

if __name__ == "__main__":
    small_bio = pickle.load(open("/lfs/furiosa/0/will/gsq/small_bio.pkl"))
    print len(small_bio[0].get_all_edges())
