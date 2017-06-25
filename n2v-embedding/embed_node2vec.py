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

def count_files(in_directory):
    joiner= (in_directory + os.path.sep).__add__
    return sum(
        os.path.isfile(filename)
        for filename
        in map(joiner, os.listdir(in_directory))
    )

def run_n2v(infile, outfile, weighted=False, verbose=False, sample=300):
    cmd = settings.N2V_DIR + " -i:" + infile + " -o:" + outfile
    if weighted:
        cmd += " -w"
    if verbose:
        cmd += " -v"
    cmd += " -s:" + str(sample)
    
    #print "CMD: " + cmd

    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=subprocess.STDOUT)

progress = tqdm(total = count_files(settings.INDIVIDUAL_UW_GRAPH_DIR)) 

for f in os.listdir(settings.INDIVIDUAL_UW_GRAPH_DIR):
    name = f.split('/')[-1].split('.')[0]
    if os.path.isfile(settings.INDIVIDUAL_UW_N2V_DIR + '/' + name + '.emb'):
        progress.update(1)
        continue
    progress.set_description("Processing " + str(name))
    try:
        run_n2v(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + f, 
            settings.INDIVIDUAL_UW_N2V_DIR + '/' + name + '.emb',
            weighted = False, verbose = True,
            sample = 100)
        progress.update(1)
    except:
        progress.update(1)
        continue
