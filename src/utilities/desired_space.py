import pandas as pd
def desired_space(desired_output, data): #entire data so x and y
    space = pd.DataFrame()
    space = data[data['class'] == desired_output] #filter instances if they have the desired y
    space = space.drop(['class'], axis=1) #drop y
    return space