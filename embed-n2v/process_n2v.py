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

from os import listdir
from os.path import isfile, join

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

directory = settings.INDIVIDUAL_UW_N2V_DIR #settings.INDIVIDUAL_UW_N2V_NODES_DIR

files = [f for f in listdir(directory) if isfile(join(directory, f))]
ids = []

N_CLUSTERS = 100
MAX_COMPONENTS = 250
HISTOGRAM = False

out_file = 'n2v-random-avg.nemb'

def get_vectors(f):
    i = open(f, 'r')
    fl = i.readline().rstrip().split(' ')
    n = int(fl[0])
    d = int(fl[1])

    v = []

    for j in range(0, min(MAX_COMPONENTS, n)):
        line = i.readline().rstrip().split(' ')
        line = [float(x) for x in line]
        v.append(line[1:])

    i.close()
    return v 

def get_histogram(f, clf):
    vv = get_vectors(directory + '/' + f)
    h = [0 for i in xrange(N_CLUSTERS)]

    for v in vv:
        v = np.array(v)
        pred = clf.predict(v.reshape(1, -1))[0]
        h[pred] += 1
    
    return h

def get_average(f):
    vv = get_vectors(directory + '/' + f)
    avg = [0] * len(vv[0])
    for v in vv:
        for i in range(0, len(v)):
            avg[i] += v[i]

    for i in range(0, len(avg)):
        avg[i] /= (len(vv) * 1.)

    return avg

km = KMeans(n_clusters = N_CLUSTERS, random_state = 0, n_jobs = -1, verbose=1)
X = []

if HISTOGRAM:

    # begin processing files

    progress = tqdm(total = len(files))
    for f in files:
        progress.set_description('File %s' % f)
        f = settings.INDIVIDUAL_UW_N2V_DIR + '/' + f 
        v = get_vectors(f)
        for t in v:
            X.append(t)

        progress.update(1)

    progress.close()

    # run kmeans

    print "[", len(X), "x", len(X[0]), "]", "Running KMeans"
    km.fit(X)
    histograms = {}

    # extract histograms

    progress = tqdm(total = len(files))
    for i in range(0, len(files)):
        progress.set_description('File %s' % files[i])
        h = get_histogram(files[i], km)
        histograms[files[i]] = h

        progress.update(1)

    progress.close()

    # save to normalized embedding file (.nemb)

    f = open(out_file, 'wb+')
    for k, v in histograms.items():
        line = str(k)[:-4]
        for val in v:
            line += (" " + str(val))

        f.write(line + "\n")

    f.close()

else:
    result = {}

    progress = tqdm(total = len(files))
    for i in range(0, len(files)):
        progress.set_description('File %s' % files[i])
        avg = get_average(files[i])
        result[files[i]] = avg

        progress.update(1)
    
    progress.close()

    # save to normalized embedding file (.nemb)

    f = open(out_file, 'wb+')
    for k, v in result.items():
        line = str(k)[:-4]
        for val in v:
            line += (" " + str(val))

        f.write(line + "\n")

    f.close()
