'''
embeds an unweighted, undirected graph with node2vec
'''
# os command
# ../common/node2vec -i:[input edgelist] -o:[output embedding] -v [verbose] -w [weighted] 

import os
import sys
import subprocess
from tqdm import tqdm

sys.path.insert(0, 'common')
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
    progress.set_description("Processing " + str(name))
    run_n2v(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + f, 
            settings.INDIVIDUAL_UW_N2V_DIR + '/' + name + '.emb',
            weighted = False, verbose = True,
            sample = 100)
    progress.update(1)

#run_n2v(settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + name + '.txt', settings.INDIVIDUAL_UW_N2V_DIR + '/' + name + '.emb', weighted=False, verbose=True, sample=100)
