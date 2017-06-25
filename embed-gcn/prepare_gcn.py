# prepares an edgelist for input to a gcn
# input: edgelist
# output: 
#   - x: the feature vectors of labeled training instances
#   - y: one hot labels of labeled training instances
#   - allx: feature vectors of labeled and unlabeled training instances
#   - graph: dict {index: [index of neighbors]}
#   - tx: feature vectors of test instances
#   - ty: one hot labels of test instances
#   - ally: labels for instances in allx

import numpy as np

# adjacency matrix
# an N by N adjacency matrix (N is the number of nodes)
def adjmat(nodes, edges):
    mat = np.zeros((len(nodes), len(nodes)))
    for edge in list(edges):
        edge = edge.split('->')
        e1 = int(edge[0])
        e2 = int(edge[1])
        mat[e1][e2] = 1
    return mat

# feature matrix
# an N by D feature matrix (D is the number of features per node)
def featmat(D):
    return np.zeros((D, D))

# binary label matrix
# an N by E binary label matrix (E is the number of classes)
def blmat(nnodes, nclasses):
    mat = np.zeros((nnodes, nclasses))
    mat[:, 0] = np.array([1]*nnodes)
    return mat

# main
f = open('test.edgelist')

nodes = set()
edges = set()

for line in f:
    line = line.split(' ')
    line = [int(x) for x in line]
    n1 = line[0]
    n2 = line[1]
    
    nodes.add(n1)
    nodes.add(n2)
    
    edge1 = str(n1) + '->' + str(n2)
    edge2 = str(n2) + '->' + str(n1)

    if edge1 in edges or edge2 in edges:
        continue
    else:
        edges.add(edge1)

print nodes
print edges
print adjmat(nodes, edges)
print featmat(10)
print blmat(len(nodes), 5)
