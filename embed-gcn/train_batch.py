from __future__ import print_function

import os    
os.environ['THEANO_FLAGS'] = "device=gpu1"    
os.environ['KERAS_BACKEND'] = "tensorflow"

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import sys
sys.path.insert(0, 'layers')

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings
import util

from graph import GraphConvolution
from utils import *

import cPickle
import numpy as np
import time
from time import gmtime, strftime

FILTER = 'chebyshev'    # 'chebyshev'
MAX_DEGREE = 2          # maximum polynomial degree
SYM_NORM = True         # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 20           # number of epochs
PATIENCE = 90           # early stopping patience

FEAT_DIM = 256         # number of randomly initialized features / node
LABELS_DIM = 3          # 2031       # number of examples

X_in = Input(shape=(FEAT_DIM,))

if FILTER == 'localpool':
    support = 1
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    support = MAX_DEGREE + 1
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

H = Dropout(0.5)(X_in)
H = GraphConvolution(32, support, activation='relu', W_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
H = GraphConvolution(16, support, activation='relu', W_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(LABELS_DIM, support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
best_val_loss = 99999

# Create list of files and associated labels
edgelists = []
for n, f in enumerate(os.listdir(settings.INDIVIDUAL_UW_GRAPH_DIR)):
    path = settings.INDIVIDUAL_UW_GRAPH_DIR + '/' + f
    edgelists.append((path, n))

edgelists = [(i[0], e) for e, i in enumerate(edgelists[0:LABELS_DIM])]

print(edgelists)

losses = []
epoch_lens = [NB_EPOCH] * len(edgelists)

# plot lists
plot_train_losses = []      # DONE
plot_test_losses = []
plot_epoch_lengths = []     # DONE
plot_epochs = []            # DONE

# OUTER LOOP: Number of Overall Epochs
for b in xrange(400):
    print("[-- EPOCH " + str(b) + " --]")
    
    # Push losses to phone
    '''
    if (b + 1) % 10 == 0:
        s = ""
        for a in losses:
            s += str(a) + " "
    
        util.push_notify(s)
    '''
    # Train more on the worst example (greater loss = train more)
    if len(losses) > 0:
        for i in range(0, len(losses)):
            epoch_lens[i] = int( losses[i] / sum(losses) * NB_EPOCH ) * len(losses)
    
    losses = []
    print("Lengths: ", epoch_lens)
    
    plot_epochs.append(b)
    plot_epoch_lengths.append(epoch_lens)

    """ INNER LOOP 1: TRAINING """

    _losses = []
    for example in xrange(len(edgelists)):

        X, A, y, n = load_edgelist(path = edgelists[example][0],
                                   label = edgelists[example][1], 
                                   labels_dim = LABELS_DIM,
                                   feat_dim = FEAT_DIM)

        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y, n)
        X = np.diag(1./np.array(X.sum(1)).flatten()).dot(X)

        if FILTER == 'localpool':
            # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) 
            A_ = preprocess_adj(A, SYM_NORM)
            support = 1
            graph = [X, A_]

        elif FILTER == 'chebyshev':
            # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016) 
            L = normalized_laplacian(A, SYM_NORM)
            L_scaled = rescale_laplacian(L)
            T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
            support = MAX_DEGREE + 1
            graph = [X]+T_k

        else:
            raise Exception('Invalid filter type.')

        print("## EXAMPLE " + str(edgelists[example][0]) + " => " + str(edgelists[example][1]))
        
        wait = 0
        tloss = None
        _losses_t = []

        for epoch in range(epoch_lens[example] + 1):

            t = time.time()

            # Single training iteration (we mask nodes without labels for loss calculation)
            model.fit(graph, y_train, sample_weight=train_mask,
                      batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

            # Predict on full dataset
            preds = model.predict(graph, batch_size=A.shape[0])

            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                           [idx_train, idx_val])
            print("\tIter: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))

            tloss = train_val_loss[1]
            if np.isinf(tloss):
                tloss = 1000 # a large number
        
            _losses_t.append([train_val_loss[0], train_val_acc[0], train_val_loss[1], train_val_acc[1]])

        losses.append(tloss)
        _losses.append(_losses_t)
    
    plot_train_losses.append(_losses)

    """ INNER LOOP 2: TESTING """

    accuracies = []
    _losses = []
    for example in xrange(len(edgelists)):
        X, A, y, n = load_edgelist(path = edgelists[example][0],
                                   label = edgelists[example][1], 
                                   labels_dim = LABELS_DIM,
                                   feat_dim = FEAT_DIM)

        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y, n)
        X = np.diag(1./np.array(X.sum(1)).flatten()).dot(X)

        if FILTER == 'localpool':
            A_ = preprocess_adj(A, SYM_NORM)
            support = 1
            graph = [X, A_]

        elif FILTER == 'chebyshev':
            L = normalized_laplacian(A, SYM_NORM)
            L_scaled = rescale_laplacian(L)
            T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
            support = MAX_DEGREE + 1
            graph = [X]+T_k

        else:
            raise Exception('Invalid filter type.')

        print("## EXAMPLE " + str(edgelists[example][0]) + " => " + str(edgelists[example][1]))
     
        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Testing
        test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
        print("\tTest set results:",
            "loss= {:.4f}".format(test_loss[0]),
            "accuracy= {:.4f}".format(test_acc[0]))
        accuracies.append(test_acc[0])
        _losses.append([test_loss[0], test_acc[0]])

    plot_test_losses.append(_losses)

    """ CHECK FOR COMPLETION """

    # all accuracies on the test set are > 0.9
    if ([i > 0.9 for i in accuracies] == [True]*len(accuracies)):
        break

""" AT COMPLETION """

directory = 'bin/' + strftime("%Y-%m-%d-%H:%M:%S", gmtime())
if not os.path.exists(directory):
    os.makedirs(directory)

model.save_weights(directory + '/model.h5')
cPickle.dump(plot_epochs, open(directory + '/epochs.p', 'wb'))
cPickle.dump(plot_epoch_lengths, open(directory + '/epoch_lengths.p', 'wb'))
cPickle.dump(plot_test_losses, open(directory + '/test_metrics.p', 'wb'))
cPickle.dump(plot_train_losses, open(directory + '/train_metrics.p', 'wb'))

util.push_notify(directory + ' => DONE') 
sys.exit(0)
