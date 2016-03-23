import numpy as np

def calc_mahalanobis(x, y, n_neighbors):
    from sklearn.neighbors import DistanceMetric, NearestNeighbors
    DistanceMetric.get_metric('mahalanobis', V=np.cov(x))

    nn = NearestNeighbors(n_neighbors=n_neighbors,
                          algorithm='brute',
                          metric='mahalanobis',
                          metric_params={'V': np.cov(x)})
    return nn.fit(x).kneighbors(y)

def calc_distance(x, y):
    return np.sum((x - y)**2, 1)

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    x, y = make_classification()
    print(calc_mahalanobis(x, x[0], 2))
