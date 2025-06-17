import random

from src.utilities.plausability import check_plausability

def SF(x,X_train,cat_f,p_num,p_cat,f,t,step):
    """
    SF permutation doesnt use MI and subspace (so also no NN from KDTree)
    param x: instance to be explained with CF
    param X_train: NOT normalized X_train (see if this is practical, if not - edit check_plausability())
    param cat_f: categorical features
    param p: user defined constrained features
    param f: black box model
    param t: target outcome
    param step: steps in how to traverse p lower and upper bounds
    returns: z, a valid countefactual instance
    """
    for i in p: #go over each feature in user defined constrained features
        z = None
        start = p[i][0] #lower bound
        end = p[i][1] #upper bound
        if i not in cat_f: #if i is not a cat feature do binary search to find the minimum value mid such that changing feature i value of x to mid will result in target outcome t and a plausible explanation z
            while start <= end: #binary search
                tempdf = x.copy()
                mid = start + (end-start)/2
                tempdf.loc[:,i] = mid
                if f.predict(tempdf)[0] == t and check_plausability(x,tempdf,X_train) == 1: #plausability through outlier detection algorithm LOF (to implement)
                    z = tempdf
                    end = mid - step[i] # try to make feature value smaller
                else:
                    start = mid + step[i] # too small, make feature value larger
        else: #if i is categorical
            z = x
            z.loc[:,i] = (1-end) #set feature i as the reverse value (check if this works with one-hot encoding)
            if f.predict(z)[0] == t and check_plausability(x,z,X_train) == 1:
                return z
    return z # if binary search for numerical feature does not succeed, returns None (no CF)

def DF(X, x, subspace, mi_pair, cat_f, num_f, features, protect_f, f, t):
    for f_pair in mi_pair:
        i = f_pair[0]
        j = f_pair[1] # i is first feature, j is second feature from tuple
        z = x # initialize CF
        if i in subspace and j in subspace:
            if (i in num_f and (j in num_f or j in cat_f)) and (i not in protect_f and j not in protect_f):
                start = subspace[i][0]
                end = subspace[i][1]
                h = regressor(X,i,j)
                g = classifier(X,i,j)
                traverse_space = sorted(random.uniform(start,end))
                while len(traverse_space) > 0:
                    mid = start + (end-start)/2
                    z.loc[:,i] = traverse_space[mid]
                    z=z.loc[:,z.columns != j]
                    if j in num_f:
                        new_j = h(z)
                        z.loc[:,j] = new_j
                    else:
                        new_j=g(z)
                        z.loc[:,j] = new_j
                    if f(z) == t and check_plausability(x,z,X) == 1:
                        return z
                    else:
                        try:
                            del traverse_space[:mid]
                        except:
                            pass

            if (i in cat_f and j in cat_f) and (i not in protect_f and j not in protect_f):
                z.loc[:,i] = subspace[i][1]
                z.loc[:,j] = subspace[j][1]
                if f(z) == t and check_plausability(x,z,X) == 1:
                    return z
    return z

#TF still needs to be done




