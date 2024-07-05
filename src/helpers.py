import numpy as np

def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))

def LDA(X, y, n_components):
    '''
    LDA essentially creates a new axis to project points to, that maximizes separability among known classes
    maximizing distance between means, and minimizing variance (or scatter)
    
    for > 3 categories, find a point that is the center of all points,
    and find the distance btwn this central point, and the central point of each category

    each LDA axis accounts for the most variation between the categories (supervised)
    '''

    pass

def PCA(n_components):
    '''
    

    each PCA axis accounts for the most variation in the data (unsupervised)
    '''

def choose_impactful_weights(self, X, y):
    pass

