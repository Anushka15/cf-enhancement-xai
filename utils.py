from sklearn.feature_selection import mutual_info_classif
def mi_score(feature1, feature2):
    return mutual_info_classif(feature1, feature2)

#
def MI (features, X):
    feature_pairs = {}
    mi_pairs = []
    for i,j in features: #features should be a list of tuples of every possible combination
        mi_scores = mi_score(X[i], X[j])
        score = mi_scores[0]
        feature_pairs[score] = (i,j) #key is score, value is pairs of features
        #potential problem is that score may be overwritten, leading to data loss
    p = dict(sorted(feature_pairs.items())) #sort the dict feature_pairs
    for i in p.keys():
        pair = p[i] # get the value (the pair of features)
        if pair not in mi_pairs:
            mi_pairs.append(pair) #return only the pairs in a list
    return mi_pairs

