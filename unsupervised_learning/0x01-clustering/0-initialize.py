import scipy
from sklearn.cluster import KMeans
import numpy as np


def initialize(X, k):
    """
    initialize: initializes cluster centroids for K-mean
    """
    n, d = X.shape

    mini = X.min(axis=0)
    maxi = X.max(axis=0)

    return np.random.uniform(mini, maxi, (k, d))
