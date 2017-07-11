# Induce subgraphs and save the representations along with labels to /dfs
from random import shuffle

# saves n subgraphs
def subgraphs(f, n):
    
    f = open(f, 'r')
    for line in f:
        line = line.split('\t')
        

# G is a NetworkX graph, n is the number of nodes
def gen_subgraph(G, n, nodes):
    shuffle(nodes)
    return G.subgraph(nodes[0:n])

# format: node1 (space) node2
def write_subgraph(G, f):
    f = open(f, 'wb+')
    for edge in G.edges():
        f.write(edge[0] + "\t" + edge[1] + "\n")
    f.close()
