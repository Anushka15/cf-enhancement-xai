import random

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.utilities.plausability import check_plausability
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def SF(x,X_train,p_num,p_cat,f,t,step):
    """
    SF permutation doesnt use MI and subspace (so also no NN from KDTree)
    param x: instance to be explained with CF
    param X_train: NOT normalized X_train (see if this is practical, if not - edit check_plausability())
    param cat_f: categorical features
    param p_num: user defined constrained numerical features
    param p_cat: user defined constrained categorical features
    param f: black box model
    param t: target outcome
    param step: steps in how to traverse p lower and upper bounds. Only applies to numerical features
    returns: z, a valid countefactual instance
    """
    print("original x prediction: ", f.predict(x))
    z_s = []
    for i in p_cat: #deze for loop eerst betekend dat cat sws verandert wordt
        z = x.copy()
        #z.loc[:,i] = p_cat[i].value() #z.loc[:,i] is select all rows of column with name i #cat zijn niet one-hot encoded hier, dus x moet niet one-hot zijn en model er ook niet op getraind zijn

        # Find all one-hot columns for feature i
        relevant_columns = [col for col in X_train.columns if col.startswith(i + "_")]

        # Set all related one-hot features to 0
        z.loc[:, relevant_columns] = 0

        # Set the specific desired one-hot feature to 1
        target_col = f"{i}_{p_cat[i]}"
        if target_col in z.columns:
            z.loc[:, target_col] = True
        else:
            print(f"all columns are 0 for this category, as it is the reference feature value")

        # kan bovenstaande ook conditioneren op onderstaande, anders niet veranderen
        #if f.predict(z)[0] == t and check_plausability(x,z,X_train) == 1:
        proba = f.predict_proba(z)[0][1]
        print(f"prediction with cat {target_col}: class={f.predict(z)[0]}, prob={proba:.4f}")
        if int(f.predict(z)[0]) == int(t):
                return z
    for i in p_num:
        print("goes in numerical, no CF based on cat found")
        start, end = p_num[i]
        print("start: ",start)
        print("end: ", end)
        while start <= end:
            tempdf = x.copy()
            mid = start + (end - start) / 2
            tempdf.loc[:, i] = mid
            proba = f.predict_proba(tempdf)[0][1]
            print(f"prediction with changed {p_num[i]}: class={f.predict(tempdf)[0]}, prob={proba:.4f}")
            # if f.predict(tempdf)[0] == t and check_plausability(x, tempdf,
            #                                                     X_train) == 1:
            if int(f.predict(tempdf)[0]) == int(t):
                z_s.append(tempdf)
                end = mid - step[i]  # try to make feature value smaller
            else:
                start = mid + step[i]  # too small, make feature value larger
    return z_s # give list for post-hoc filtering # or return last item of list, since this has the smallest change from binary search? # i think binary search is minimizing the feature value, not nessecarily the minimal change to x, this can also be a critique point

# def SF(x,X_train,cat_f,p,f,t,step):
#     """
#     SF permutation doesnt use MI and subspace (so also no NN from KDTree)
#     param x: instance to be explained with CF
#     param X_train: NOT normalized X_train (see if this is practical, if not - edit check_plausability())
#     param cat_f: categorical features
#     param p: user defined constrained features
#     param f: black box model
#     param t: target outcome
#     param step: steps in how to traverse p lower and upper bounds
#     returns: z, a valid countefactual instance
#     """
#     for i in p: #go over each feature in user defined constrained features
#         z = None
#         start = p[i][0] #lower bound
#         end = p[i][1] #upper bound
#         if i not in cat_f: #if i is not a cat feature do binary search to find the minimum value mid such that changing feature i value of x to mid will result in target outcome t and a plausible explanation z
#             while start <= end: #binary search
#                 tempdf = x.copy()
#                 mid = start + (end-start)/2
#                 tempdf.loc[:,i] = mid
#                 if f.predict(tempdf)[0] == t and check_plausability(x,tempdf,X_train) == 1: #plausability through outlier detection algorithm LOF (to implement)
#                     z = tempdf
#                     end = mid - step[i] # try to make feature value smaller
#                 else:
#                     start = mid + step[i] # too small, make feature value larger
#         else: #if i is categorical
#             z = x
#             z.loc[:,i] = (1-end) #set feature i as the reverse value (check if this works with one-hot encoding)
#             if f.predict(z)[0] == t and check_plausability(x,z,X_train) == 1:
#                 return z
#     return z # if binary search for numerical feature does not succeed, returns None (no CF) #this is likely another fault in the algorithm from the paper, because it keeps looking, which has this risk of overwriting for numerical features

def DF(df, y_train, x, subspace, mi_pair, cat_f, num_f, features, protect_f, f, t):
    # does not use p-map
    for f_pair in mi_pair:
        i = f_pair[0]
        j = f_pair[1] # i is first feature, j is second feature from tuple
        z = x.copy() # initialize CF
        if i in subspace and j in subspace:
            if (i in num_f and j in num_f) and (i not in protect_f and j not in protect_f):
                start = subspace[i][0]
                end = subspace[i][1]
                h = regressor(df,i,j)
                traverse_space = sorted(random.uniform(start,end))
                while len(traverse_space) > 0:
                    mid = start + (end-start)/2
                    z.loc[:,i] = traverse_space[mid]
                    #z=z.loc[:,z.columns != j]
                    new_j = h(z)
                    z.loc[:,j] = new_j
                    if f(z) == t: #and check_plausability(x,z,X) == 1:
                        return z
                    else:
                        try:
                            del traverse_space[:mid] # make the space smaller
                        except:
                            pass

            elif (i in cat_f and j in num_f) and (i not in protect_f and j not in protect_f):
                start = subspace[i][0] #if cat, then only has 1 value
                #end = subspace[i][1]
                h = regressor(df, i, j)
                while len(traverse_space) > 0:
                    mid = start + (end - start) / 2
                    z.loc[:, i] = traverse_space[mid]
                    # z=z.loc[:,z.columns != j]
                    new_j = h(z)
                    z.loc[:, j] = new_j
                    if f(z) == t:  # and check_plausability(x,z,X) == 1:
                        return z
                    else:
                        print('can not find CF for this cat value i, going to next feature pair')

            elif (i in num_f and j in cat_f) and (i not in protect_f and j not in protect_f):
                start = subspace[i][0]
                end = subspace[i][1]
                g = classifier(df, i, j)
                traverse_space = sorted(random.uniform(start, end))
                while len(traverse_space) > 0:
                    mid = start + (end - start) / 2
                    z.loc[:, i] = traverse_space[mid]
                    z = z.loc[:, z.columns != j]
                    if j in num_f:
                        new_j = h(z)
                        z.loc[:, j] = new_j
                    else:
                        new_j = g(z)
                        z.loc[:, j] = new_j
                    if f(z) == t and check_plausability(x, z, X) == 1:
                        return z
                    else:
                        try:
                            del traverse_space[:mid]  # make the space smaller
                        except:
                            pass

            elif (i in cat_f and j in cat_f) and (i not in protect_f and j not in protect_f):
                g = classifier(df, i, j)
                z.loc[:,i] = subspace[i][1]
                z.loc[:,j] = subspace[j][1]
                if f(z) == t and check_plausability(x,z,X) == 1:
                    return z
    return z

#TF still needs to be done

def regressor(df, f_j):
    # train regressor to predict feature j based on i from traverse space
    """
    :param df: dataframe of all the data
    :param f_j: the feature that we want to predict, the 'y'
    note: 'class' is now also a independent feature
    :return:
    """
    X = np.array(df.loc[:, df.columns != f_j])
    y = np.array(df.loc[:, df.columns == f_j])
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42) # moet ook preprocessed? ja denk het wel om preprocessed waarden te voorspellen
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train.ravel()) # .ravel() is
    y_pred = linear_reg.predict(X_test)
    from sklearn.metrics import mean_squared_error
    import math
    mse = mean_squared_error(y_test, y_pred)
    msse = math.sqrt(mean_squared_error(y_test, y_pred))
    return linear_reg, mse, msse

def classifier(df, f_j):
    # train classifier to predict feature j based on i from traverse space
    X = np.array(df.loc[:, df.columns != f_j])
    y = np.array(df.loc[:, df.columns == f_j])
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42) # moet ook preprocessed? ja denk het wel om preprocessed waarden te voorspellen
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train.ravel()) # .ravel() is
    y_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return log_reg, acc





