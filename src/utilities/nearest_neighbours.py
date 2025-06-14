import pandas as pd
from scipy.spatial import KDTree

def FNN(desired_space,x,radius):
    """
    Creates a KDTree object for finding the region for perturbations
    param desired_space: the feature space to consider (a collection of feature vectors). It is a subset of the training data which contains only positive outcome-based data instances.
    param x: instance for which we want to find neighbours/perturbations
    param radius: to measure the distance from x to a point
    returns: the indices of the nearest neighbours found, these are possible perturbations, not the final ones, because MI is used for that as well.
    """
    tree = KDTree(desired_space)
    idx = tree.query_ball_point(x,r=radius)
    nn = tree.data[idx]
    nn = pd.DataFrame.from_records(nn, columns=x.columns)
    return nn

def intervals(nn, p, f2change, x):
    """
    Define a subspace (upper and lower bounds) according to user set constraints (p), possible perturbations (nn) and user set constraints (f2change)
    param nn: possible perturbations
    param p: dictionary with features with lower and upper bound of the user-specified interval for ith feature. (e.g. income range)
    returns: subspace
    """
    subspace = {}
    for i in p:
        lower = p[i][0]
        upper = p[i][1]
        if upper >= nn[i].max():
            subspace[i] = [lower, nn[i].max()]
        elif lower <= nn[i].min():
            subspace[i] = [nn[i].min(), upper]
        else:
            subspace[i] = [lower, upper]
    return subspace