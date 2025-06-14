import pandas as pd
def desired_space(desired_output, data): #entire data so x and y
    """
    Creates feature space of instances with desired output
    param desired_output: y, either 1 or 2
    param data: dataframe including input X and output y
    returns: feature space of only feature vectors with y=desired_output
    """
    space = pd.DataFrame()
    space = data[data['class'] == desired_output] #filter instances if they have the desired y
    space = space.drop(['class'], axis=1) #drop y
    return space