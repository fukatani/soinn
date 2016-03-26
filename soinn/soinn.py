# Copyright (c) 2016 Yoshihiro Nakamura
# Copyright (c) 2016 Ryosuke Fukatani
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import numpy as np
from scipy.sparse import dok_matrix
import util


class Soinn(object):
    """ Self-Organizing Incremental Neural Network (SOINN)
        Ver. 0.1.0
    """

    def __init__(self, delete_node_period=300, max_edge_age=50):
        """
        :param delete_node_period: A period deleting nodes.
                The nodes that doesn't satisfy some condition are deleted every this period.
        :param max_edge_age: The maximum of edges' ages.
                If an edge's age is more than this, the edge is deleted.
        :return:
        """
        self.delete_node_period = delete_node_period
        self.max_edge_age = max_edge_age
        self.min_degree = 1
        self.num_signal = 0
        self.nodes = np.array([], dtype=np.float64)
        self.winning_times = []
        self.adjacent_mat = dok_matrix((0, 0), dtype=np.float64)

    def input_signal(self, signal, learning=True):
        """ Input a new signal to SOINN
        :param signal: A new input signal
        :return:
        """
        self.__check_signal(signal)
        self.num_signal += 1

        if self.nodes.shape[0] < 3:
            self.__add_node(signal)
            return

        winner, dists = self.__find_nearest_nodes(2, signal)

        if not learning:
            return winner

        sim_thresholds = self.calculate_similarity_thresholds(winner)
        if dists[0] > sim_thresholds[0] and dists[1] > sim_thresholds[1]:
            self.__add_node(signal)
        else:
            self.__add_edge(winner)
            self.__increment_edge_ages(winner[1])
            winner[1] = self.__delete_old_edges(winner[1])
            self.__update_winner(winner[1], signal)
            self.__update_adjacent_nodes(winner[1], signal)

        if self.num_signal % self.delete_node_period == 0:
            self.__delete_noise_nodes()
        return winner

    def __check_signal(self, signal):
        """ check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as self.dim.
        So, this method have to be called before calling functions that use self.dim.
        :param signal: an input signal
        """
        if not(isinstance(signal, np.ndarray)):
            raise TypeError()
        if len(signal.shape) != 1:
            raise TypeError()
        if not(hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                raise TypeError()

    def __add_node(self, signal):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim))
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))

    def __find_nearest_nodes(self, num, signal, mahar=True):
        #if mahar: return self.__find_nearest_nodes_by_mahar(num, signal)
        n = self.nodes.shape[0]
        indexes = [0.0] * num
        sq_dists = [0.0] * num
        D = util.calc_distance(self.nodes, np.asarray([signal] * n))
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    def __find_nearest_nodes_by_mahar(self, num, signal):
        indexes, sq_dists = util.calc_mahalanobis(self.nodes, signal, 2)
        return indexes, sq_dists

    def calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0:
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:
                pal_indexes = []
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = util.calc_distance(self.nodes[pal_indexes], np.asarray([self.nodes[i]] * len(pal_indexes)))
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    def __add_edge(self, node_indexes):
        self.__set_edge_weight(node_indexes, 1)

    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    def __delete_old_edges(self, winner_index):
        candidates = []
        for k, v in self.adjacent_mat[winner_index, :].items():
            if v > self.max_edge_age + 1:
                candidates.append(k[1])
                self.__set_edge_weight((winner_index, k[1]), 0)
        delete_indexes = []
        for i in candidates:
            if len(self.adjacent_mat[i, :]) == 0:
                delete_indexes.append(i)
        self.__delete_nodes(delete_indexes)
        delete_count = sum([1 if i < winner_index else 0 for i in delete_indexes])
        return winner_index - delete_count

    def __set_edge_weight(self, index, weight):
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    def __update_winner(self, winner_index, signal):
        self.winning_times[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w)/self.winning_times[winner_index]

    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w)/(100 * self.winning_times[i])

    def __delete_nodes(self, indexes):
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        if indexes:
            self.winning_times = [self.winning_times[i] for i in remained_indexes]
            self.adjacent_mat = self.adjacent_mat[np.ix_(remained_indexes, remained_indexes)]

    def __delete_nodes2(self, indexes):
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        if indexes:
            self.winning_times = [self.winning_times[i] for i in remained_indexes]
            self.adjacent_mat = self.adjacent_mat[np.ix_(remained_indexes, remained_indexes)]

    def __delete_noise_nodes(self):
        n = len(self.winning_times)
        noise_indexes = []
        for i in range(n):
            if len(self.adjacent_mat[i, :]) < self.min_degree:
                noise_indexes.append(i)
        self.__delete_nodes(noise_indexes)

    def print_info(self):
        print('Total Nodes: {0}'.format(len(self.nodes)))

    def save(self, dumpfile='soinn.dump'):
        import joblib
        joblib.dump(self, dumpfile, compress=True, protocol=0)
