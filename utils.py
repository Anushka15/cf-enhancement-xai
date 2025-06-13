import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.spatial import KDTree
def mi_score(feature1, feature2):
    return mutual_info_classif(feature1, feature2)

def MI(features, X):
    """
    param features: should be a list of tuples of every possible combination of feature pairs
    param X: data
    returns: mi_pairs, a list of tuples of feature pairs, sorted on from highest to lowest MI
    """
    feature_pairs = {}
    mi_pairs = []
    for i,j in features: #features should be a list of tuples of every possible combination
        mi_scores = mi_score(X[i], X[j])
        score = mi_scores[0]
        feature_pairs[score] = (i,j) #key is score, value is pairs of features
        #potential problem is that score may be overwritten, leading to data loss
    p = dict(sorted(feature_pairs.items(), reverse=True)) #sort the dict feature_pairs
    for i in p.keys():
        pair = p[i] # get the value (the pair of features)
        if pair not in mi_pairs:
            mi_pairs.append(pair) #return only the pairs in a list
    return mi_pairs

def FNN(desired_space,x,radius):
    """
    Creates a KDTree object for finding the nearest neighbour efficiently
    param desired_space: consider only the partitions from the desired neighbourhood from user constraints
    param x: instance for which we want to find neighbours
    param radius: to measure the distance from x to a point
    returns: the indices of the nearest neighbours found
    """
    tree = KDTree(desired_space)
    idx = tree.query_ball_point(x,r=radius)
    nn = tree.data[idx]
    nn = pd.DataFrame.from_records(nn, columns=x.columns)
    return nn

def intervals(nn, perturb_map, f2change, x):
    """
    param nn:
    param perturb_map:
    param radius: to restrict the neighbourhood around x
    returns: subspace
    """
    subspace = {}
    for i in perturb_map:
        lower = p[i][0]
        upper = p[i][0]
        if upper >= nn[i].max():
            subspace[i] = [lower, nn[i].max()]
        elif lower <= nn[i].min():
            subspace[i] = [nn[i].min(), upper]
        else:
            subspace[i] = [lower, upper]
    return subspace
