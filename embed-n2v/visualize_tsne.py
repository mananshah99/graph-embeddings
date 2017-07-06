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

#set up NCBI database
from ete2 import NCBITaxa
ncbi = NCBITaxa('/dfs/scratch0/manans/.etetoolkit/taxa.sqlite')

# read .nemb file
EMBEDDING_FILE = 'emb/n2v-kmeans.nemb'
histograms = []
species_ids = []

with open(EMBEDDING_FILE, 'r') as tf:
    for line in tf:
        ls = line.split(' ')
        species_ids.append(ls[0])
        v = []
        for n in ls[1:]:
            v.append(float(n))
        histograms.append(v)

from mgkit.taxon import *

print "Reading UniprotTaxonomy from NCBI dump"
mgk = UniprotTaxonomy()
mgk.read_from_ncbi_dump('/dfs/scratch0/manans/ncbi/nodes.dmp', '/dfs/scratch0/manans/ncbi/names.dmp', '/dfs/scratch0/manans/ncbi/merged.dmp')

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
                
#print id_to_domain('394'), id_to_x('394', 'class'), id_to_i('394', 2)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    color_idx = 0
    for i in range(X.shape[0]):
        domain = id_to_domain(species_ids[i])
        text_str = id_to_i(species_ids[i], 2)
        
        clr = 'k'
        
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
        '''

        point = plt.plot(X[i, 0], X[i, 1], color=clr, marker='o')
        label = ['Domain: ' + str(id_to_domain(species_ids[i])) + ' / Subkingdom: ' + str(id_to_i(species_ids[i], 2)) + ' [' + species_ids[i] + ']' ]
        tooltip = mpld3.plugins.PointLabelTooltip(point[0], labels=label)
        mpld3.plugins.connect(fig, tooltip)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# tSNE

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
