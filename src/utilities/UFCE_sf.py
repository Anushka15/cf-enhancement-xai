from src.utilities.plausability import check_plausability

def SF(x,X_train,cat_f,p,f,t,step):
    """
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
        start = p[i][0] #lower bound
        end = p[i][1] #upper bound
        if i not in cat_f: #if i is not a cat feature do binary search to find the minimum value mid such that changing feature i value of x to mid will result in target outcome t and a plausible explanation z
            while start <= end: #binary search
                tempdf = x.copy()
                mid = start + (end-start)/2
                tempdf.loc[:,i] = mid
                if f(tempdf) == t and check_plausability(x,tempdf,X_train) == 1: #plausability through outlier detection algorithm LOF (to implement)
                    z = tempdf
                    end = mid - step[i] # try to make feature value smaller
                else:
                    start = mid + step[i] # too small, make feature value larger
        else: #if i is categorical
            z = x
            z.loc[:,i] = (1-end) #set feature i as the reverse value (check if this works with one-hot encoding)
            if f(z) == t and check_plausability(x,z,X_train) == 1:
                return z
    return z
