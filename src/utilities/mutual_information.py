from sklearn.feature_selection import mutual_info_regression


def mi_score(feature1, feature2):
    return mutual_info_regression(feature1.values.reshape(-1, 1), feature2, discrete_features='auto')[0] #not truly symmetric, but also used by the authors,
    # feature 2 is here seen as a 'target', hence no only feature1 is reshaped

def MI(features, X):
    """
    param features: should be a list of tuples of every possible combination of feature pairs
    param X: data
    returns: mi_pairs, a list of tuples of feature pairs, sorted on from highest to lowest MI
    """
    feature_pairs = {}
    mi_pairs = []
    for f1, f2 in features: #features should be a list of tuples of every possible combination
        mi_scores = mi_score(X[f1], X[f2])
        score = mi_scores[0]
        feature_pairs[score] = (f1, f2) #key is score, value is pairs of features
        #potential problem is that score may be overwritten, leading to data loss
    p = dict(sorted(feature_pairs.items(), reverse=True)) #sort the dict feature_pairs
    for i in p.keys():
        pair = p[i] # get the value (the pair of features)
        if pair not in mi_pairs:
            mi_pairs.append(pair) #return only the pairs in a list
    return mi_pairs