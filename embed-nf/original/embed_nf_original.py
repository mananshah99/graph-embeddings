import os, pickle
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('..')

# Neural Fingerprint
from neuralfingerprint import load_data, relu
from neuralfingerprint import build_conv_deep_net, build_convnet_fingerprint_fun
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad, degrees, build_standard_net
from neuralfingerprint.util import rmse
from neuralfingerprint.data_util import remove_duplicates

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/graph-embeddings/common/')
import settings
import util

def count_files(in_directory):
    joiner = (in_directory + os.path.sep).__add__
    return sum(
            os.path.isfile(filename)
            for filename
            in map(joiner, os.listdir(in_directory))
            )

normalize = 0
params = {'fp_length': 50, # 20,
            'fp_depth': 3,
            'init_scale':np.exp(-4),
            'learn_rate':np.exp(-9),
                    'b1':np.exp(-4),
                    'b2':np.exp(-4),
            'l2_penalty':np.exp(-4),
            'l1_penalty':np.exp(-5),
            'conv_width':10}

conv_layer_sizes = [params['conv_width']] * params['fp_depth']
conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                    'fp_length' : params['fp_length'],
                    'normalize' : normalize,
                    'return_atom_activations':False}

def embed_nf_original(input_directory, 
                      output_directory, 
                      weights='/afs/cs.stanford.edu/u/manans/graph-embeddings/embed-nf/original/results.pkl',
                      verbose = False,
                      overwrite = True):

    trained_weights = None
    with open(weights) as f:
        trained_weights = pickle.load(f)

    conv_arch_params['return_atom_activations'] = True
    output_layer_fun, parser, compute_atom_activations = \
       build_convnet_fingerprint_fun(**conv_arch_params)
   
    progress = tqdm(total = count_files(input_directory), disable = not verbose)
    files = np.asarray([input_directory + i for i in os.listdir(input_directory)])
    embeddings = output_layer_fun(trained_weights, np.asarray(files))

    output_map = {}
    for i, f in enumerate(files):
        output_map[f.split('/')[-1].split('.')[0]] = embeddings[i]
        with open(output_directory + '/' + f.split('/')[-1].split('.')[0] + '.nemb', 'wb+') as f2:
            for q in embeddings[i]:
                f2.write(str(q))
                f2.write(' ')
