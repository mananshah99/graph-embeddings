from __future__ import print_function

import theano
import time
import numpy as np
import sklearn.metrics as metrics
import sys
import warnings
from keras.models import Model
import keras.backend as backend
from theano import function

import neuralfingerprint.utils as utils
import neuralfingerprint.data_preprocessing as data_preprocessing
import neuralfingerprint.fingerprint_model_matrix_based as fingerprint_model_matrix_based
import neuralfingerprint.fingerprint_model_index_based as fingerprint_model_index_based

from matplotlib import pyplot

def lim(float, precision = 5):
    return ("{0:."+str(precision)+"f}").format(float)

def save_model_visualization(model, filename='model.png'):
    try:
        from keras.utils.visualize_util import plot
        plot(model, filename, show_shapes=1)
    except:
        import traceback
        print('\nsave_model_visualization() failed with exception:',traceback.format_exc())

def predict(data, model):
    pred = []    
    for batch in data:
        if len(batch)==2:
            batch = batch[0]
        pred.append(model.predict_on_batch(batch))
    return np.concatenate(pred)

def eval_metrics_on(predictions, labels):
    if len(labels[0])==2: #labels is list of data/labels pairs
        labels = np.concatenate([l[1] for l in labels])
    predictions = predictions[:,0]
    
    r2                       = metrics.r2_score(labels, predictions)
    mean_abs_error           = np.abs(predictions - labels).mean()
    mse                      = ((predictions - labels)**2).mean()
    rmse                     = np.sqrt(mse)
    median_absolute_error    = metrics.median_absolute_error(labels, predictions) # robust to outliers
    explained_variance_score = metrics.explained_variance_score(labels, predictions) # best score = 1, lower is worse
    return {'r2':r2, 'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse, 
            'median_absolute_error':median_absolute_error, 
            'explained_variance_score':explained_variance_score}

def parity_plot(predictions, labels):
    try:
        figure = pyplot.figure()
    except:
        print('parity_plot:: Error: Cannot create figure')
        return
    ax  = figure.add_subplot(111)
    ax.set_axisbelow(True)
    
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Predicted', fontsize=15)
    pyplot.grid(b=True, which='major', color='lightgray', linestyle='--')
    pyplot.title('Parity Plot')
    pyplot.scatter(labels, predictions, s=15, c='b', marker='o')
    
def test_on(data, model, description='test_data score:'):
    scores=[]
    weights =[]
    for v in data:
        weights.append(v[1].shape) # size of batch
        scores.append( model.test_on_batch(x=v[0], y=v[1]))
    weights = np.array(weights)
    s=np.mean(np.array(scores)* weights/weights.mean())
    if len(description):
        print(description, lim(s))
    return s

def get_model_params(model):
    weight_values = []
    for lay in model.layers:
        weight_values.extend( backend.batch_get_value(lay.weights))
    return weight_values

def set_model_params(model, weight_values):
    symb_weights = []
    for lay in model.layers:
        symb_weights.extend(lay.weights)
    assert len(symb_weights) == len(weight_values)
    for model_w, w in zip(symb_weights, weight_values):
        backend.set_value(model_w, w)
        
def save_model_weights(model, filename = 'model.npz'):
    ws = get_model_params(model)
    np.savez(filename, ws)

def load_model_weights(model, filename = 'model.npz'):
    ws = np.load(filename)
    set_model_params(model, ws[ws.keys()[0]])
    
def update_lr(model, initial_lr, relative_progress, total_lr_decay):
    """
    initial_lr: any float (most reasonable values are in the range of 1e-5 to 1)
    total_lr_decay: value in (0, 1] -- this is the relative final LR at the end of training
    relative_progress: value in [0, 1] -- current position in training, where 0 = beginning, 1 = end of training
    """
    assert total_lr_decay > 0 and total_lr_decay <= 1
    backend.set_value(model.optimizer.lr, initial_lr * total_lr_decay**(relative_progress))
    

def train_model(model, train_data, valid_data, test_data, 
                 batchsize = 100, num_epochs = 100, train = True, 
                 initial_lr=3e-3, total_lr_decay=0.2, verbose = 1):
    if train:
        log_train_mse = []
        log_validation_mse = []
        best_valid = 9e9
        model_params_at_best_valid=[]
        
        times=[]
        for epoch in range(num_epochs):
            update_lr(model, initial_lr, epoch*1./num_epochs, total_lr_decay)
            batch_order = np.random.permutation(len(train_data))
            losses=[]
            t0 = time.clock()
            for i in batch_order:

                loss = model.train_on_batch(x=train_data[i][0], y=train_data[i][1], check_batch_dim=False)
                losses.append(loss)

                model_part = Model(input=model.input, output=model.layers[-5].output)
                embeddings = model_part.predict_on_batch(train_data[i][0])

                '''
                try:
                    print("EXPECTED INPUT", model.input)
                    print("ACTUAL INPUT", train_data[i][0].keys())
                   
                    exp_keys = []
                    for expected in model.input:
                        exp_keys.append(str(expected))

                    for k in train_data[i][0].keys():
                        if k not in exp_keys:
                            print("EXTRANEOUS", k)
                            del train_data[i][0][k]

                    print("EXPECTED INPUT", model.input)
                    print("ACTUAL INPUT", train_data[i][0].keys())
                   
                    for key in exp_keys:
                        if key not in train_data[i][0].keys():
                            print("MISSING KEY", key)

                    get_activations = backend.function([model.input, backend.learning_phase()], model.layers[-5].output)
                    activations = get_activations(train_data[i][0], 1)
                    sys.exit(-1)

                    print(activations)

                except Exception as e:
                    print(e)
                    continue
                '''

            times.append(time.clock()-t0)
            val_mse = test_on(valid_data,model,'valid_data score:' if verbose>1 else '')
            if best_valid > val_mse:
                best_valid = val_mse
                model_params_at_best_valid = get_model_params(model) #kept in RAM (not saved to disk as that is slower)
            if verbose>0:
                print('Epoch',epoch+1,'completed with average loss',lim(np.mean(losses)))
            log_train_mse.append(np.mean(losses))
            log_validation_mse.append(val_mse)
            
        print('Training @',lim(1./np.mean(times[1:])),'epochs/sec (',lim(batchsize*len(train_data)/np.mean(times[1:])),'examples/s)')
    
    set_model_params(model, model_params_at_best_valid)
    
    training_data_scores   = eval_metrics_on(predict(train_data,model), train_data)
    validation_data_scores = eval_metrics_on(predict(valid_data,model), valid_data)
    test_predictions = predict(test_data,model)
    test_data_scores       = eval_metrics_on(test_predictions, test_data)
    
    print('training set mse (best_val):  ', lim(training_data_scores['mse']))
    print('validation set mse (best_val):', lim(validation_data_scores['mse']))
    print('test set mse (best_val):      ', lim(test_data_scores['mse']))
    
    return model, (training_data_scores, validation_data_scores, test_data_scores), (log_train_mse, log_validation_mse), test_predictions

def plot_training_mse_evolution(data_lists, legend_names=[], ylabel = 'MSE', xlabel = 'training epoch', legend_location='best'):
    
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    try:
        figure = pyplot.figure()
    except:
        print('plot_training_mse_evolution:: Error: Cannot create figure')
        return
    ax  = figure.add_subplot(111)
    ax.set_axisbelow(True)
    if len(legend_names):
        assert len(legend_names)==len(data_lists), 'you did not provide enough or too many labels for the graph'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    pyplot.grid(b=True, which='major', color='lightgray', linestyle='--')
    if len(legend_names) != len(data_lists):
        legend_names = [' ' for x in data_lists]
    for i, data in enumerate(data_lists):
        assert len(data)==len(data_lists[0])
        pyplot.plot(np.arange(1,len(data)+1), data, 
                    _colors[i%len(_colors)], linestyle='-', marker='o', 
                    markersize=5, markeredgewidth=0.5, linewidth=2.5, label=legend_names[i])
    if len(legend_names[0]):
        ax.legend(loc=legend_location, shadow=0, prop={'size':14}, numpoints=1)
    
def cv(use_matrix_based_implementation = True, plot_training_mse = False, train = True):
    
    np.random.seed(1338)  
    num_epochs = 10
    batchsize  = 1 #20   #batch size for training
    L2_reg     = 4e-3
    batch_normalization = 0
    
    fp_length = 51  # size of the final constructed fingerprint vector
    conv_width = 50 # number of filters per fingerprint layer
    fp_depth = 3    # number of convolutional fingerprint layers
    
    n_hidden_units = 100
    predictor_MLP_layers = [n_hidden_units, n_hidden_units, n_hidden_units]    
    
    # total number of cross-validation splits to perform
    crossval_total_num_splits = 3
    
    data = []
    labels = []
    with open('input.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            data.append(line.split(' ')[0])
            labels.append(float(line.split(' ')[1]))
    data = np.array(data)
    labels = np.array(labels)

    print(data[0], labels[0])
    print('# of valid examples in data set:',len(data))

    train_mse = []
    val_mse   = []
    test_mse  = []
    test_scores = []
    all_test_predictions = []
    all_test_labels = []
    
    if use_matrix_based_implementation:
        fn_build_model = fingerprint_model_matrix_based.build_fingerprint_regression_model
    else:
        fn_build_model = fingerprint_model_index_based.build_fingerprint_regression_model
    
    print('Mean | MSE = ', lim(np.mean((labels-labels.mean())**2)), ' and RMSE =', lim(np.sqrt(np.mean((labels-labels.mean())**2))))
    
    for crossval_split_index in range(crossval_total_num_splits):
        print('CV // Split', crossval_split_index+1,'of',crossval_total_num_splits)
    
        traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=crossval_split_index, 
                                                                    crossval_total_num_splits=crossval_total_num_splits, 
                                                                    validation_data_ratio=0.1)

        train, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(traindata, valdata, testdata, 
                                                                     training_batchsize = batchsize, 
                                                                     testset_batchsize = 1000)
        
        print("Preprocessed dataset")

        model = fn_build_model(fp_length = fp_length, fp_depth = fp_depth, 
                               conv_width = conv_width, predictor_MLP_layers = predictor_MLP_layers, 
                               L2_reg = L2_reg, num_input_atom_features = 62, 
                               num_bond_features = 6, batch_normalization = batch_normalization)

        model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch, test_predictions = \
            train_model(model, train, valid_data, test_data, batchsize = batchsize, num_epochs = num_epochs, train=1)

        train_mse.append(train_scores_at_valbest['mse'])
        val_mse.append(val_scores_best['mse'])
        test_mse.append(test_scores_at_valbest['mse'])
        test_scores.append(test_scores_at_valbest)
        all_test_predictions.append(test_predictions[:,0])
        all_test_labels.append(np.concatenate(map(lambda x:x[-1],test_data)))
        
        if plot_training_mse:
            plot_training_mse_evolution(train_valid_mse_per_epoch, ['training set MSE (+regularizer)', 'validation set MSE'])
            pyplot.draw()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pyplot.pause(0.0001)
    
    parity_plot(np.concatenate(all_test_predictions), np.concatenate(all_test_labels))
    
    print('\n\nCrossvalidation complete!\n')
    
    print('Mean training_data MSE =', lim(np.mean(train_mse)), '+-', lim(np.std(train_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean validation    MSE =', lim(np.mean(val_mse)), '+-', lim(np.std(val_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data     MSE =', lim(np.mean(test_mse)), '+-', lim(np.std(test_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data     RMSE =', lim(np.mean(np.sqrt(np.array(test_mse)))), '+-', lim(np.std(np.sqrt(np.array(test_mse)))/np.sqrt(crossval_total_num_splits)))
    
    avg_test_scores = np.array([x.values() for x in test_scores]).mean(0)
    avg_test_scores_dict = dict(zip(test_scores[0].keys(), avg_test_scores))
    print()
    for k,v in avg_test_scores_dict.items():
        if k not in ['mse','rmse']:
            print('Test-set',k,'=',lim(v))
                
    return model
    
if __name__=='__main__':
    
    plot_training_mse = 0
    
    model = cv(use_matrix_based_implementation=0, plot_training_mse = plot_training_mse, train=True)
    #save_model_weights(model, 'trained_fingerprint_model.npz')
   
    '''
    # to load the saved model weights use e.g.:
    load_model_weights(model, 'trained_fingerprint_model.npz')

    data = []
    labels = []
    with open('input.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            data.append(line.split(' ')[0])
            labels.append(float(line.split(' ')[1]))
    data = np.array(data)
    labels = np.array(labels)

    print(data[0], labels[0])
    print('# of valid examples in data set:',len(data))

    traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=1, 
                                                                    crossval_total_num_splits=2, 
                                                                    validation_data_ratio=0.1)
       

    train, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(traindata, valdata, testdata, 
                                                                     training_batchsize = 1, testset_batchsize=1) 
   
    X = np.array(train)
    print(model.predict(x=X[0][0], batch_size = 1, verbose=1)) 
    #print(model.summary())
    #print(model.layers[-5].get_weights())

    #save_model_visualization(model, filename = 'fingerprintmodel.png')
    '''
