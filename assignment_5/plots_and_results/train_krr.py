def train_krr(X_train, Y_train,kwidth,llambda):
    ''' Trains kernel ridge regression (krr)
    Input:       X_train  -  DxN array of N data points with D features
                 Y        -  D2xN array of length N with D2 multiple labels
                 kwdith   -  kernel width
                 llambda    -  regularization parameter
    Output:      alphas   -  NxD2 array, weighting of training data used for apply_krr                     
    '''
    # your code here
    K = GaussianKernel(X_train, X_train, kwidth)
    KLI = (K + np.diag(llambda * np.ones(len(K))))
    alphas = np.linalg.inv(KLI).dot(Y_train.T)
    return alphas
