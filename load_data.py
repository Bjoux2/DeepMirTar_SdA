import cPickle as pk
import numpy
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
import theano
import theano.tensor as T
base_path = '/home/thl/my_task/for-data3/'

def load_data(rand_seed=1235):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    f = open(base_path + 'data3/pkl/CLASH_N_features_dic_all.pkl')
    CLASH_N_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/CLASH_features_dic_all.pkl')
    CLASH_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/Mark_features_dic_all.pkl')
    Mark_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/Mark_N_features_dic_all.pkl')
    Mark_N_features_dic = pk.load(f)
    f.close()


    X_p = []
    for key in CLASH_features_dic.keys():
        X_p.append(CLASH_features_dic[key].values())
    for key in Mark_features_dic.keys():
        X_p.append(Mark_features_dic[key].values())

    X_n = []
    for key in CLASH_N_features_dic.keys():
        X_n.append(CLASH_N_features_dic[key].values())
    for key in Mark_N_features_dic.keys():
        X_n.append(Mark_N_features_dic[key].values())

    X = np.array(X_p + X_n)
    y = np.array([1]*len(X_p) + [0]*len(X_n))

    np.random.seed(rand_seed)
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)
    Y = y[p]

    return X, Y
    # min_max_scaler = MinMaxScaler()  ## min max scaler
    # X = min_max_scaler.fit_transform(X)
    # print X.shape
    # print y.shape
    #
    #
    # train_X = X[:5000]
    # val_X = X[5000:6500]
    # test_X = X[6500:]
    #
    # train_Y = Y[:5000]
    # val_Y = Y[5000:6500]
    # test_Y = Y[6500:]
    #
    # train_set = (train_X, train_Y)
    # valid_set = (val_X, val_Y)
    # test_set  = (test_X, test_Y)
    # # train_set, valid_set, test_set format: tuple(input, target)
    # # input is a numpy.ndarray of 2 dimensions (a matrix)
    # # where each row corresponds to an example. target is a
    # # numpy.ndarray of 1 dimension (vector) that has the same length as
    # # the number of rows in the input. It should give the target
    # # to the example with the same index in the input.
    #
    # def shared_dataset(data_xy, borrow=True):
    #     """ Function that loads the dataset into shared variables
    #
    #     The reason we store our dataset in shared variables is to allow
    #     Theano to copy it into the GPU memory (when code is run on GPU).
    #     Since copying data into the GPU is slow, copying a minibatch everytime
    #     is needed (the default behaviour if the data is not in a shared
    #     variable) would lead to a large decrease in performance.
    #     """
    #     data_x, data_y = data_xy
    #     shared_x = theano.shared(numpy.asarray(data_x,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     shared_y = theano.shared(numpy.asarray(data_y,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     # When storing data on the GPU it has to be stored as floats
    #     # therefore we will store the labels as ``floatX`` as well
    #     # (``shared_y`` does exactly that). But during our computations
    #     # we need them as ints (we use labels as index, and if they are
    #     # floats it doesn't make sense) therefore instead of returning
    #     # ``shared_y`` we will have to cast it to int. This little hack
    #     # lets ous get around this issue
    #     return shared_x, T.cast(shared_y, 'int32')
    #
    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)
    #
    # rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
    #         (test_set_x, test_set_y)]
    # return rval


if __name__ == '__main__':
    load_data()