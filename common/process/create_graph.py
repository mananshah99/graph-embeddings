import os
import sys
import numpy as np
from tqdm import tqdm
import networkx as nx
import pygraphviz as pgv
import re

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings
import util

# G = nx.Graph()
# INDIVIDUAL_PPI_DIR/*.txt
# 882

def create_graph(name, weighted=False):
    i = open(settings.INDIVIDUAL_PPI_DIR + '/' + name + '.txt', 'r')
    if weighted:
        o = open(settings.INDIVIDUAL_W_GRAPH_DIR + '/' + name + '.txt', 'wb')
    else:
        o = open(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + name + '.txt', 'wb')
    
    nd = re.compile(r'[^\d.]+')
    for line in tqdm(i, total=util.count_lines(settings.INDIVIDUAL_PPI_DIR + '/' + name + '.txt')):
        try:
            v = line.split(' ')
            w = int(v[2]) + int(v[3]) + int(v[4]) + int(v[5]) + int(v[6]) + int(v[7])
            n1 = int(nd.sub('', v[0].split('.')[1]).lstrip('0'))
            n2 = int(nd.sub('', v[1].split('.')[1]).lstrip('0'))
    
            if weighted and w > 0:
                o.write(str(n1) + ' ' + str(n2) + ' ' + str(w) + '\n')
            if not weighted and w > 0:
                o.write(str(n1) + ' ' + str(n2) + '\n')
        except:
            print "ERROR - ", line
            pass
    o.close()

def visualize_graph(graph_path, output = ''):
    weighted = False
    if settings.INDIVIDUAL_W_GRAPH_DIR in graph_path:
        weighted = True

    G = pgv.AGraph()
    for line in tqdm(open(graph_path, 'r'), total = util.count_lines(graph_path)):
        v = line.split(' ')
        n1 = v[0]
        n2 = v[1]
        G.add_edge(n1, n2)
        if weighted:
            edge = G.get_edge(n1, n2)
            edge.attr['weight'] = int(v[2])

    
    if output == '':
        output = graph_path.split('/')[-1].strip('.txt') + '.dot'
    
    G.write(output)

for f in tqdm(os.listdir(settings.INDIVIDUAL_PPI_DIR), total=2031):
    n = f.split('/')[-1].split('.')[0]
    create_graph(n, weighted=False)
    create_graph(n, weighted=True)

#create_graph('882', weighted=True)
#visualize_graph(settings.INDIVIDUAL_W_GRAPH_DIR + '/882.txt')
