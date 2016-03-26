# This is a example script for clastering MNIST using soinn.
# Learning and evaluate learn result.
# If the same number in the same cluster is a lot, the result is precise.
#
# Copyright (c) 2016 Ryosuke Fukatani
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import numpy as np
from soinn import Soinn

from sklearn.datasets import fetch_mldata
from collections import defaultdict

def prepare_dataset():
    print('load MNIST dataset')
    mnist = fetch_mldata('MNIST original')
    mnist['data'] = mnist['data'].astype(np.float32)
    mnist['data'] /= 255
    mnist['target'] = mnist['target'].astype(np.int32)
    return mnist

def learning(soinn, x_train, y_train):
    for i, data_x in enumerate(x_train):
        if i % 1000 == 0:
            print('Processing {0}th data.'.format(i))
        soinn.input_signal(data_x)

def evaluate(soinn, x_test, y_test, N_test=2000):
    #N_test = y_test.size
    answer_dict = defaultdict(list)
    indexes = [i for i in range(len(soinn.nodes))]
    similarity_thresholds = soinn.calculate_similarity_thresholds(indexes)
    for i in range(N_test):
        winner = soinn.input_signal(x_test[i], learning=False)
        answer_dict[winner[0]].append(y_test[i])
    for node_num, value in answer_dict.items():
        print('node[{0}] radius: {1}'.format(node_num, similarity_thresholds[node_num]))
        print('values:{0}'.format(value))

def split_dataset(dataset, N=4000):
    perm = np.random.permutation(len(dataset['target']))
    dataset['data'] = dataset['data'][perm]
    dataset['target'] = dataset['target'][perm]
    x_train, x_test = np.split(dataset['data'],   [N])
    y_train, y_test = np.split(dataset['target'], [N])
    return x_train, y_train, x_test, y_test

def visualise(soinn_nodes):
    import matplotlib.pyplot as plt
    MAX_PLOT = 100
    COLUMN = 10
    for i, node in enumerate(soinn_nodes):
        if i == MAX_PLOT: break
        draw_digit(node, i+1, MAX_PLOT / COLUMN, COLUMN, "ans")
    plt.show()

def draw_digit(data, n, row, col, title):
    import matplotlib.pyplot as plt
    size = 28
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,28)
    plt.ylim(0,28)
    plt.pcolor(Z)
    plt.title("title=%s"%(title), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

if __name__ == '__main__':
    dataset = prepare_dataset()
    N_TRAIN = 5000
    delete_node_period = 100
    max_edge_age = 30
    x_train, y_train, x_test, y_test = split_dataset(dataset, N_TRAIN)
    dumpfile = 'soinn{0}_{1}_{2}.dump'.format(N_TRAIN,
                                              delete_node_period,
                                              max_edge_age)
    try:
        import joblib
        soinn_i = joblib.load(dumpfile)
    except:
        print('New SOINN is created.')
        soinn_i = Soinn(delete_node_period=delete_node_period,
                        max_edge_age=max_edge_age)
        learning(soinn_i, x_train, y_train)
    evaluate(soinn_i, x_test, y_test)
    #visualise(soinn_i.nodes)
    soinn_i.print_info()
    soinn_i.save(dumpfile)

