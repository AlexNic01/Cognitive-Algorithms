
def train_lda(X,Y):
    ''' Trains a linear discriminant analysis
    Definition:  w, b   = train_lda(X,Y)
    Input:       X       -  DxN array of N data points with D features
                 Y       -  1D array of length N of class labels {-1, 1}
    Output:      w       -  1D array of length D, weight vector  
                 b       -  bias term for linear classification                          
    '''
    # your code here 
    # hint: use the scipy/numpy function sp.cov
    w_p = X.T[Y==1].mean(axis=0)
    w_m = X.T[Y==-1].mean(axis=0)
    n = len(w_p)
    S_B = (w_p - w_m).reshape(n,1).dot((w_p - w_m).reshape(1,n))
    S_W = np.cov(X.T[Y==1], rowvar = False, bias = 1)
    S_W += np.cov(X.T[Y==-1], rowvar = False, bias = 1)
    return np.dot(np.linalg.inv(S_W), (w_p - w_m)), 0
