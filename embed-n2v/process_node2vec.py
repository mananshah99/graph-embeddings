'''
processes node2vec embeddings

input: embedding file [# embeddings, # dimensions]
steps:
    -- for all files -- 
    1. separate each vector (length = # dimensions)
    2. add the vector to a general KMeans clustering framework
    -- for each file --
    3. create a histogram of length D = number of dimensions in KMeans
    4. increment frequencies in histogram corresponding to labels of each vector
    5. save histogram as representation of the graph
'''
import os
import sys
from tqdm import tqdm
import numpy as np
from time import time

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings
import util

from sklearn.cluster import KMeans

from os import listdir
from os.path import isfile, join

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt, mpld3
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

import six
from matplotlib import colors

# Set up colors for plotting
colors_ = list(six.iteritems(colors.cnames))
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

hex_ = [color[1] for color in colors_]
rgb = [colors.hex2color(color) for color in hex_]
hsv = [colors.rgb_to_hsv(color) for color in rgb]

hue = [color[0] for color in hsv]
sat = [color[1] for color in hsv]
val = [color[2] for color in hsv]

ind = np.lexsort((val, sat, hue))
sorted_colors = [colors_[i] for i in ind]
colors_final = []

for i, (name, color) in enumerate(sorted_colors):
    colors_final.append(color)

import random
random.shuffle(colors_final)

colors_mapping = {}

from ete2 import NCBITaxa

#set up NCBI database
ncbi = NCBITaxa('/dfs/scratch0/manans/.etetoolkit/taxa.sqlite')

files = [f for f in listdir(settings.INDIVIDUAL_UW_N2V_NODES_DIR) if isfile(join(settings.INDIVIDUAL_UW_N2V_NODES_DIR, f))]
files = files[0:500]
ids = []

N_CLUSTERS = 75 #150
GRAPH = 0

# Fill the ids array
with open(settings.SPECIES_MAPPING, 'r') as f:
    for line in f:
        ids.append(line.rstrip().split('\t'))

# Initialize MGKit
from mgkit.taxon import *

print "Reading UniprotTaxonomy from NCBI dump"
mgk = UniprotTaxonomy()
mgk.read_from_ncbi_dump('/dfs/scratch0/manans/ncbi/nodes.dmp', '/dfs/scratch0/manans/ncbi/names.dmp', '/dfs/scratch0/manans/ncbi/merged.dmp')

def id_to_mapping(id_str):
    for idx in ids:
        # find the id
        if id_str == idx[0]:
            return idx[2]

def id_to_domain(id_str):
    lineage = ncbi.get_lineage(id_str)
    try:
        domain_map = ncbi.get_taxid_translator([lineage[2]])
        return domain_map[lineage[2]]
    except:
        domain_map = ncbi.get_taxid_translator([lineage[0]])
        return domain_map[lineage[0]]

def id_to_x(id_str, x):
    lineage = ncbi.get_lineage(id_str)
    ranks_tmp = ncbi.get_rank(lineage)
    ranks = [str(ranks_tmp[j]) for j in lineage]
    #print ranks, lineage
    idx = 0
    for r in ranks:
        if r == x:
            idx = ranks.index(x)
            break

    domain_map = ncbi.get_taxid_translator([lineage[idx]])
    return domain_map[lineage[idx]]

def id_to_i(id_str, i):
    try:
        lineage = get_lineage(mgk, float(id_str), names=True)
        return lineage[i]
    except:
        return id_to_x(id_str, 'domain') 
                
def tree(id_str):
    t = ncbi.get_topology([float(id_str)], intermediate_nodes=True)
    print t.get_ascii(attributes=["sci_name"])

#print id_to_domain('394'), id_to_x('394', 'class'), id_to_i('394', 2)
#print id_to_domain('10090'), id_to_x('10090', 'class'), id_to_i('10090', 2)
#print tree('394')

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    color_idx = 0
    #ax = plt.subplot(111)
    for i in range(X.shape[0]):
        domain = id_to_domain(str(files[i])[:-4])
        text_str = domain #id_to_i(str(files[i])[:-4], 2)  #id_to_x(str(files[i])[:-4], 'class')
        
        clr = 'k'
        '''
        if domain == 'Bacteria':
            clr = 'b'
        elif domain == 'Archaea':
            clr = 'r'
        elif domain == 'Eukaryota':
            clr = 'y'
        '''

        if text_str in colors_mapping:
            clr = colors_final[colors_mapping[text_str]]
        
        elif color_idx < len(colors_final):
            colors_mapping[text_str] = color_idx
            clr = colors_final[colors_mapping[text_str]]
            color_idx += 1
            print "Using color", color_idx
        
        point = plt.plot(X[i, 0], X[i, 1], color=clr, marker='o')
        '''
        plt.text(X[i, 0], X[i, 1], text_str,
                 color=clr,
                 fontdict={'weight': 'bold', 'size': 9})
        '''
        label = ['Domain: ' + str(id_to_domain(str(files[i])[:-4])) + ' / Subkingdom: ' + str(id_to_i(str(files[i])[:-4], 2)) + ' [' + str(files[i])[:-4] + ']']
        tooltip = mpld3.plugins.PointLabelTooltip(point[0], labels=label)
        mpld3.plugins.connect(fig, tooltip)

    '''
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    '''
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def get_vectors(f):
    i = open(f, 'r')
    fl = i.readline().rstrip().split(' ')
    n = int(fl[0])
    d = int(fl[1])

    v = []

    for j in range(0, n):#min(100,n)):#min(200,n)):#n):
        line = i.readline().rstrip().split(' ')
        line = [float(x) for x in line]
        v.append(line[1:])

    i.close()
    return v 

def get_histogram(f, clf):
    vv = get_vectors(settings.INDIVIDUAL_UW_N2V_DIR + '/' + f)
    h = [0 for i in xrange(N_CLUSTERS)]

    for v in vv:
        v = np.array(v)
        pred = clf.predict(v.reshape(1, -1))[0]
        h[pred] += 1
    
    return h

## KMEANS

km = KMeans(n_clusters = N_CLUSTERS, random_state = 0, n_jobs = -1, verbose=1)
X = []

progress = tqdm(total = len(files))

for f in files:
    progress.set_description('File %s' % f)
    f = settings.INDIVIDUAL_UW_N2V_DIR + '/' + f 
    v = get_vectors(f)
    for t in v:
        X.append(t)

    progress.update(1)

progress.close()

if GRAPH:
    
    # 1 -- elbow plot
    print "Method 1 -- Elbow Plot"
    K = range(50, 100, 10)
    KM = [KMeans(n_clusters = k).fit(X) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    print "\t Obtained centroids"

    D_k = [cdist(dt_trans, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d)/dt_trans.shape[0] for d in dist]

    print "\t Obtained avgWithinSS"

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(dt_trans)**2)/dt_trans.shape[0]
    bss = tss-wcss
    
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('KMeans Cluster Elbow Plot')

    plt.savefig('elbow-ss.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('KMeans Cluster Elbow Plot')

    plt.savefig('elbow-ver.png')

    #2 -- silhouette plot
    print "Method 2 -- Silhouette Plot"

    s = []
    for n_clusters in range(2,30):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(dt_trans)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        s.append(silhouette_score(dt_trans, labels, metric='euclidean'))

        plt.plot(s)
        plt.ylabel("Silhouette Score")
        plt.xlabel("k")
        plt.title("Silhouette Score for K-means cell's behavior")

    plt.savefig('silhouette.png')

print len(X), len(X[0]), "Running KMeans"

km.fit(X)

histograms = []
for i in tqdm(range(0, len(files))):
    h = get_histogram(files[i], km)
    histograms.append(h)

print("Computing t-SNE embedding")
print("Histogram is of length " + str(len(histograms)) + " with " + str(len(histograms[0])) + " dimensions / histogram")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=35, n_iter=3000, verbose=2)
t0 = time()
X_tsne = tsne.fit_transform(histograms)

plot_embedding(X_tsne,
               "t-SNE embedding of the histograms (time %.2fs)" %
               (time() - t0))
mpld3.save_html(plt.gcf(), 'tsne.html')
plt.savefig('tsne.png')

plt.figure()
x_plt = []
y_plt = []

for i in tqdm(range(0, len(files[0:50]))):
    for j in tqdm(range(len(files[0:50]))):
        if i == j:
            continue
        h_i = get_histogram(files[i], km)
        h_j = get_histogram(files[j], km)
        dist_i_j = np.linalg.norm(np.array(h_i) - np.array(h_j))
        dist_taxa = distance_two_taxa(mgk, float(str(files[i])[:-4]), float(str(files[j])[:-4]))
        #print i, j, dist_i_j, dist_taxa
        x_plt.append(dist_i_j)
        y_plt.append(dist_taxa)

plt.scatter(x_plt, y_plt)
print x_plt
print y_plt

mpld3.save_html(plt.gcf(), 'scatter.html')
plt.savefig('scatter.png')
