'''
embeds an unweighted, undirected graph with node2vec
'''
import os
import sys
import subprocess
from tqdm import tqdm

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings
import util
import numpy as np

def count_files(in_directory):
    joiner= (in_directory + os.path.sep).__add__
    return sum(
        os.path.isfile(filename)
        for filename
        in map(joiner, os.listdir(in_directory))
    )

def run_n2v(infile, outfile, weighted=False, verbose=False, sample=300, nodes=""):
    cmd = settings.N2V_DIR + " -i:" + infile + " -o:" + outfile
    if weighted:
        cmd += " -w"
    if verbose:
        cmd += " -v"
    if nodes != "":
        cmd += " -f:" + nodes
    cmd += " -s:" + str(sample)
    
    #print "CMD: " + cmd
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=subprocess.STDOUT)

""" ORIGINAL EXPERIMENT CODE

    progress = tqdm(total = count_files(settings.INDIVIDUAL_UW_GRAPH_DIR)) 

    for f in os.listdir(settings.INDIVIDUAL_UW_GRAPH_DIR):
        name = f.split('/')[-1].split('.')[0]
        if os.path.isfile(settings.INDIVIDUAL_UW_N2V_NODES_DIR + '/' + name + '.emb'): #TODO: Change this line
            progress.update(1)
            continue
        progress.set_description("Processing " + str(name))
        try:
            # round 1 -- random sample 100 nodes
            '''
            run_n2v(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + f, 
                settings.INDIVIDUAL_UW_N2V_DIR + '/' + name + '.emb',
                weighted = False, verbose = True,
                sample = 100)
            '''
            # round 2 -- only select ribosomal nodes
            run_n2v(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + f,
                    settings.INDIVIDUAL_UW_N2V_NODES_DIR + '/' + name + '.emb',
                    weighted = False, verbose = True,
                    sample = -1, nodes = settings.INDIVIDUAL_UW_NODES_DIR + '/' + f)
            progress.update(1)
        except Exception as e:
            print e
            progress.update(1)
            continue
"""

def embed_n2v(input_directory, # = settings.INDIVIDUAL_UW_GRAPH_DIR,
              output_directory, # = settings.INDIVIDUAL_UW_N2V_NODES_DIR,
              weighted = False,
              verbose = False,
              sample = -1,
              selected_nodes = "",
              overwrite = True): # = settings.INDIVIDUAL_UW_NODES_DIR 

    progress = tqdm(total = count_files(input_directory), disable = not verbose) 

    for f in os.listdir(input_directory):
        name = f.split('/')[-1].split('.')[0]
        if os.path.isfile(output_directory + '/' + name + '.emb') and not overwrite:
            progress.update(1)
            continue
        progress.set_description("Processing " + str(name))
        try:
            run_n2v(input_directory + '/' + f,
                    output_directory + '/' + name + '.emb',
                    weighted = False, verbose = True,
                    sample = sample, nodes = selected_nodes + '/' + f if selected_nodes != "" else "")
            progress.update(1)
        except Exception as e:
            print e
            progress.update(1)
            continue
    
    output_map = {}
    for f in os.listdir(output_directory):
        if f.endswith('.nemb'):
            continue
        name = f.split('/')[-1].split('.')[0]
        fl = open(output_directory + '/' + f, 'r')
        v = []
        length = -1
        length2 = -1
        for n, line in enumerate(fl.readlines()):
            if n == 0:
                length = int(line.split(' ')[1])
                length2 = int(line.split(' ')[0])
                v = [0] * length
                continue
            else:
                vv = [float(i) for i in line.split(' ')[1:]]
                for i in xrange(len(vv)):
                    v[i] += vv[i]
        
        v = [i / float(length2) for i in v]
        with open(output_directory + '/' + name + '.nemb', 'wb+') as f2:
            for i in v:
                f2.write(str(i))
                f2.write(' ')
        output_map[name] = v

    return output_map
