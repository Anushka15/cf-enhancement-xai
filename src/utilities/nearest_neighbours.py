import pandas as pd
from scipy.spatial import KDTree

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