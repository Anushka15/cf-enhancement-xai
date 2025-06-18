import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def check_plausability(x,z,X):
    """
    Check if CF is plausible based on Local Outlier Factor (LOF) Algorithm
    param x: test instance you want to explain with CF
    param X: NOT normalized training data (see if this is practical, if not- remove scaler) # now it gets normalized
    param z: CF of x
    returns: 1 or 0, representing if CF is plausible or not
    """
    # scaler = StandardScaler()
    # scaler = scaler.fit(X)
    # n_X = scaler.transform(X)
    # n_x = scaler.transform(x)
    # n_z = scaler.transform(z)
    X_tot = pd.concat([X, x], ignore_index=True)

    clf = LocalOutlierFactor(n_neighbors=100, novelty=True)
    clf.fit(X_tot)
    no_outlier = clf.predict(z)
    return no_outlier # 1 if no outlier, else 0

