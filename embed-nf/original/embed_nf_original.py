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
sys.path.append('../embed-nf/original/')

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

task_params = {'N_train'     : 35,
               'N_valid'     : 10,
               'N_test'      : 10,
               'target_name' : 'label',
               'data_file'   : 'rewire.csv'}

num_epochs = 5
batch_size = 10 #100
dropout = 0
activation = relu
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

def parse_training_params(params):
    nn_train_params = {'num_epochs'  : num_epochs,
                       'batch_size'  : batch_size,
                       'learn_rate'  : params['learn_rate'],
                       'b1'          : params['b1'],
                       'b2'          : params['b2'],
                       'param_scale' : params['init_scale']}

    vanilla_net_params = {'layer_sizes' : [params['fp_length']],  # Linear regression.
                          'normalize'   : normalize,
                          'L2_reg'      : params['l2_penalty'],
                          'L1_reg'      : params['l1_penalty'],
                          'activation_function':activation}
    return nn_train_params, vanilla_net_params

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    npr.seed(0)
    init_weights = npr.randn(num_weights) * train_params['param_scale']
    train_targets, undo_norm = normalize_array(train_raw_targets)

    training_curve = []
    def callback(weights, iter):
        if True or iter % 10 == 0:
            # print "max of weights", np.max(np.abs(weights))
            train_preds = undo_norm(pred_fun(weights, train_smiles))
            cur_loss = loss_fun(weights, train_smiles, train_targets)
            training_curve.append(cur_loss)

            print "Iteration", iter, "loss", cur_loss, "train RMSE", \
                np.sqrt(np.mean((train_preds - train_raw_targets)**2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", iter, ":", \
                    np.sqrt(np.mean((validation_preds - validation_raw_targets) ** 2)),
            print ""

        if len(training_curve) > 2 and training_curve[-2] < training_curve[-1]:
            train_params['learn_rate'] /= 10.
            print "\t Updated learning rate =>", train_params['learn_rate']
        else:
            print "\t Learning rate is constant =>", train_params['learn_rate']
        
        return train_params['learn_rate']
        
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    num_iters = train_params['num_epochs'] * len(train_smiles) / train_params['batch_size']
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=num_iters, step_size=train_params['learn_rate'],
                           b1=train_params['b1'], b2=train_params['b2'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

def train_neural_fingerprint(train_directory, labels_mapping, tmp_dir):
    global task_params
    task_params['N_train'] = int(len(os.listdir(train_directory)) * 0.7)
    task_params['N_valid'] = int(len(os.listdir(train_directory)) * 0.1)
    task_params['N_test']  = int(len(os.listdir(train_directory)) * 0.2)
    task_params['data_file'] = tmp_dir

    directory = train_directory
    output = open(tmp_dir, 'wb+')

    files = os.listdir(directory)
    output.write('graph,label\n')
    for f in files:
        output.write(directory + '/' +  f + ',' + str(labels_mapping[f]) + '\n')
    output.close()
    
    print "Loading data..."
    traindata, valdata, testdata = load_data(task_params['data_file'],
                        (task_params['N_train'], task_params['N_valid'], task_params['N_test']),
                        input_name='graph', target_name=task_params['target_name'])
    train_inputs, train_targets = traindata
    val_inputs, val_targets = valdata

    print "Regression on", task_params['N_train'], "training points."
    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", rmse(train_preds, train_targets)
        print "Test: ", rmse(val_preds,  val_targets)
        print "-" * 80
        return rmse(val_preds,  val_targets)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean)

    print "Task params", params
    nn_train_params, vanilla_net_params = parse_training_params(params)
    conv_arch_params['return_atom_activations'] = False

    loss_fun, pred_fun, conv_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params, params['l2_penalty'])
    num_weights = len(conv_parser)

    predict_func, trained_weights, conv_training_curve = \
         train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                 nn_train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)

    print_performance(predict_func)
    return trained_weights

def train_nf_original(train_directory, labels_mapping, tmp_dir, output_directory):
    trained_network_weights = train_neural_fingerprint(train_directory, labels_mapping, tmp_dir)

    with open(output_directory, 'wb+') as f:
        pickle.dump(trained_network_weights, f)
 
def embed_nf_original(input_directory, 
                      output_directory, 
                      weights,
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
        progress.update(1)
    progress.close()

    return output_map
