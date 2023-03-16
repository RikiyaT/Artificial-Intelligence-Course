class KNearestNeighborsRegressor:
    # K: the number of training data points joining the voting
    def __init__(self, K):
        self.K = K

    # fit: train this model on training inputs X and outputs Y
    # X: training inputs -- np.ndarray
    #      (shape: [# of data points, # of features])
    # Y: training outputs -- np.ndarray
    #      (shape: [# of data points])
    def fit(self, X, y):
        X, y = check_X_y(X, y)
         # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

        pass

    # predict: classify given data points
    # X: inputs to the classifier -- np.ndarray
    #      (shape: [# of data points, # of features])
    def predict(self, X):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        closest = numpy.sqrt( ((X_ - X) ** 2.).sum(axis=1) )
        return self.y_[closest]

        # Hint: Euclid distances between the training inputs X_ and
        #   prediction inputs X with shape [# of data points, # of features] are
        #   calculuated by ``numpy.sqrt( ((X_ - X) ** 2.).sum(axis=1) )``
        pass

from sklearn.neighbors import KNeighborsRegressor as KNearestNeighborsRegressor

# check this is a main file
if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from pandas import DataFrame
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor as KNearestNeighborsRegressor
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    boston = load_boston()
    boston_df = DataFrame(boston.data)
    boston_df.columns = boston.feature_names
    X_train, X_test, Y_train, Y_test = train_test_split(boston_df, boston.target)
    knr = KNearestNeighborsRegressor(5)
    knr.fit(X_train, Y_train)
    Y_pred = knr.predict(X_test)

    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    print('Mean Absolute Error:', mae)
    print('Root Mean Square Error:', rmse)
