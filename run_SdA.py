from __future__ import print_function

import os
import sys
import timeit

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn import metrics

from Stack_dA import SdA


def run_SdA(X, Y,
            preprocessing = 'MinMaxScaler',
            train_valid_test_ratio=[0.6, 0.2, 0.2],
            hidden_layers_sizes=[1000, 1000, 1000, 1000, 1000],
            n_outs=2,
            finetune_lr=0.01,
            pretraining_epochs=100,
            pretrain_lr=0.001,
            training_epochs=1000,
            batch_size=5):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    # # data preprocessing
    if preprocessing == 'MinMaxScaler':
        min_max_scaler = MinMaxScaler()  # # min max scaler
        X = min_max_scaler.fit_transform(X)
    elif preprocessing == 'Normalizer':
        normalize_scaler = Normalizer(norm='l2')  # l1, l2, or max
        X = normalize_scaler.fit_transform(X)
    elif preprocessing == 'StandardScaler':
        standard_scaler = StandardScaler()  # l1, l2, or max
        X = standard_scaler.fit_transform(X)


    # # data shuffle
    np.random.seed(1234)
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)
    Y = Y[p]

    # # data split
    shape0 = X.shape[0]
    r1 = int(train_valid_test_ratio[0]*shape0)
    r2 = int((train_valid_test_ratio[0]+train_valid_test_ratio[1])*shape0)

    train_set_x = X[0:r1]
    valid_set_x = X[r1:r2]
    test_set_x = X[r2:]

    train_set_y = Y[:r1]
    valid_set_y = Y[r1:r2]
    test_set_y = Y[r2:]

    # # data shared for GPU process
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=train_set_x.get_value(borrow=True).shape[1],
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=n_outs
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [0.1]*5+[0.15]*5+[0.2]*20
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c, dtype='float64')))

    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        train_set_x=train_set_x,
        train_set_y=train_set_y,
        valid_set_x=valid_set_x,
        valid_set_y=valid_set_y,
        test_set_x=test_set_x,
        test_set_y=test_set_y,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # # Wen Ming: test predprab on the test set
                    # train_predprob = train_predprob_model()
                    # train_predclass = train_predclass_model()
                    # valid_predprob = valid_predprob_model()
                    # valid_predclass = valid_predclass_model()
                    # test_predprob = test_predprob_model()
                    # test_predclass = test_predclass_model()

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    # # Wen Ming
    # finetune_time = (end_time - start_time) / 60.
    # y_train_true = np.array(train_set_y.eval())
    # y_valid_true = np.array(valid_set_y.eval())
    # y_test_true = np.array(test_set_y.eval())
    #
    #
    # auc_train = metrics.roc_auc_score(y_train_true, train_predprob[:, 1])
    # acc_train = metrics.accuracy_score(y_train_true, train_predclass)
    # tpr_train = metrics.recall_score(y_train_true, train_predclass)
    # tnr_train = (acc_train * y_train_true.shape[0] - list(y_train_true).count(1) * tpr_train)/list(y_train_true).count(0)
    #
    # auc_valid = metrics.roc_auc_score(y_valid_true, valid_predprob[:, 1])
    # acc_valid = metrics.accuracy_score(y_valid_true, valid_predclass)
    # tpr_valid = metrics.recall_score(y_valid_true, valid_predclass)
    # tnr_valid = (acc_valid * y_valid_true.shape[0] - list(y_valid_true).count(1) * tpr_valid)/list(y_valid_true).count(0)
    #
    # auc_test = metrics.roc_auc_score(y_test_true, test_predprob[:, 1])
    # acc_test = metrics.accuracy_score(y_test_true, test_predclass)
    # tpr_test = metrics.recall_score(y_test_true, test_predclass)
    # tnr_test = (acc_test * y_test_true.shape[0] - list(y_test_true).count(1) * tpr_test)/list(y_test_true).count(0)
    #
    # return auc_train, acc_train, tpr_train, tnr_train,auc_valid, acc_valid, tpr_valid, tnr_valid,auc_test, acc_test, tpr_test, tnr_test, pretrain_time, finetune_time

    return best_validation_loss, test_score

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


if __name__ == '__main__':
    run_SdA()
