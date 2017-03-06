# e-commerce project

import pandas as pd
import numpy as np

def getData():
    
    df=pd.read_csv("F:/PYTHON PROGRAMS/gitClone/machine_learning_examples-master/ann_logistic_extra/ecommerce_data.csv")
    #print(df.head())
    data=df.as_matrix()

    #creating x(input) and y(required output)

    x=data[:,:-1] #last coloumn is action
    y=data[:,-1]

     # normalize columns 1 and 2
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] # non-categorical

   # one-hot
    for n in xrange(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
        
    return X2, Y

def get_binary_data():
    # return only the data from the first 2 classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
    
