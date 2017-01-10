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
    

if __name__ == '__main__':
    load_data()
