import numpy as np
import math
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable
from gsq.model import Encoder
import sys

def make_random_graphs(num_graphs, nodes_per_graph, density):
    """
    Generate a set of random graphs.
    As a hack, these are all disconnected components of one large graph.
    Returns:
        adj_lists -- map from dummy relation -> node id -> list of neighbours
        graph_nodes -- map from graph id -> set of nodes that belong to that graph
    """

    graph_nodes = {}

    # The graphs are single-layer, so just one mode and one relation
    adj_lists = {("mode1", "rel1", "mode1") : defaultdict(list)} 
    for i in range(num_graphs):
        graph_node_ids = range(i*nodes_per_graph, (i+1)*nodes_per_graph)
        graph_nodes[i] = set(graph_node_ids)

        # add random neighbors per node
        for node in graph_node_ids:
            adj_lists[node] = np.random.choice(graph_node_ids, 
                    size=int(math.floor(nodes_per_graph*density)), 
                    replace=False)
    return adj_lists, graph_nodes

def read_edgelist(edgelists):
    graph_nodes = {}
    adj_lists = {("mode1", "rel1", "mode1") : defaultdict(list)}

    for idx, edgelist in enumerate(edgelists):
        nodes = set()
        curr = 0
        with open(edgelist, 'r') as f:
            for line in f:
                line = line.strip('\n').split(' ')
                
                # shift by 'curr' so that we differentiate graphs in A
                nodes.add(curr + int(line[0]))
                nodes.add(curr + int(line[1]))
                adj_lists[curr + int(line[0])] = curr + int(line[1])
                adj_lists[curr + int(line[1])] = curr + int(line[0])

        graph_nodes[idx] = nodes
        curr += len(list(nodes))

    return adj_lists, graph_nodes

class MaxPoolingAggregator(nn.Module):
    """
    Module that aggregates sets of nodes and makes predictions with
    a cross-entropy classification loss.
    Aggregation is done with a single-layer MLP + max pooling
    """

    def __init__(self, node_encoder, 
            hidden_dim=64, num_classes=2):
        super(MaxPoolingAggregator, self).__init__()
        self.node_encoder = node_encoder
        self.dummy_mode = node_encoder.relations.keys()[0]
        self.layer1 = nn.Linear(node_encoder.out_dims[self.dummy_mode], hidden_dim)
        self.non_lin = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_list):
        node_embeds = self.node_encoder(node_list, self.dummy_mode) 
        out = self.non_lin(self.layer1(node_embeds.t()))
        out = out.max(dim=0)[0]
        out = self.layer2(out.unsqueeze(0))
        return out, node_embeds

def train():
    #adj_lists, graph_nodes = make_random_graphs(100, 10, 0.1)

    inputs = ['test.edgelist', 'test2.edgelist']
    labels = [0, 1]
    adj_lists, graph_nodes = read_edgelist(inputs)
    
    features = lambda nodes, mode, offset : Variable(torch.FloatTensor(np.random.random(size=(len(nodes), 16))))
    feature_dims = {"mode1" : 16}
    out_dims = {"mode1" : 16}
    relations = {"mode1" : [("mode1", "rel1")]}

    ## Now we make a two-level convolutional encoder...
    #enc1 corresponds to K=1
    enc1 = Encoder(features, 
            feature_dims, 
            out_dims, 
            relations, 
            adj_lists)
    #enc2 corresponds to k=2
    enc2 = Encoder(lambda nodes, mode, offset: 
            enc1.forward(nodes, mode, offset).t(),
            enc1.out_dims, 
            out_dims,
            relations, 
            adj_lists)

    graph_classifier = MaxPoolingAggregator(enc2)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(graph_classifier.parameters(), lr=0.01)
    for train_it in range(2):
        print "Epoch ", train_it
        for train_index in range(len(inputs)):
            optimizer.zero_grad()
            
            graph = graph_nodes[train_index]
            label = Variable(torch.LongTensor(np.array([labels[train_index]])))
            output = graph_classifier.forward(graph)[0]
        
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    
    test_adj_lists, test_nodes = read_edgelist(inputs)
    print graph_classifier.forward(test_nodes[1])[1], test_nodes[1]

if __name__ == "__main__":
    train()
