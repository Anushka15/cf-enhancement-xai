import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def check_plausability(x,z,X):
    """
    Check if CF is plausible based on Local Outlier Factor (LOF) Algorithm
    param x: test instance you want to explain with CF
    param z: CF of x
    param X: training data
    returns: 1 or 0, representing if CF is plausible or not
    """
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    n_X = scaler.transform(X)
    n_x = scaler.transform(x)
    n_z = scaler.transform(z)

    clf = LocalOutlierFactor(n_neighbors=100, novelty=True)
    clf.fit(n_X)
    no_outlier = clf.predict(n_z)
    return no_outlier # 1 if no outlier, else 0

