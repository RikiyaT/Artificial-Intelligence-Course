import numpy as np
class KNNRegression:
    """
    Methods:
    -------
    fit: Calculate distances and ranks based on given data
    predict: Predict the K nearest self.neighbors based on problem type
    """ 
    import numpy as np

    def __init__(self, k):
        """
            Parameters
            ----------
            k: Number of nearest self.neighbors
            distance_metric: Distance metric to be used. (Euclidean)
        """
        self.k = k
        self.distance_metric = self.euclidean

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        import numpy as np

        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []

        # Calculating distances  
        for i in range(n):  # for every sample in X_test
            distance = []  # To store the distances
            for j in range(m):  # for every sample in X_train 
                distance.append((self.distance_metric, y_train[j]))    
            distance = sorted(distance) # sorting distances in ascending order

            # Getting k nearest neighbors
            neighbors = []
            for item in range(self.k):
                neighbors.append(distance[item][1])  # appending K nearest neighbors

            # Making predictions
            y_pred.append(np.mean(neighbors))  # For Regression 
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
        
    def euclidean(self, point, data):
        '''Euclidean distance between a point  & data'''
        return np.sqrt(np.sum((point - data)**2, axis=1))
    
    
if __name__ == "__main__":
    from sklearn.datasets import load_boston

    dataset = load_boston()

    x = dataset.data
    y = dataset.target
    # Loading a function to split a dataset
    from sklearn.model_selection import train_test_split
    # Split into training and testing data sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # defining model
    reg_model = KNNRegression(3)

    # training model
    reg_model.fit(x_train, y_train)

    reg_model.predict(x_test[:1])

    print(reg_model.evaluate(x_test, y_test))

# This code is using knn regression.In kNN regression, we look at k training data near the data and take 
# the average of those training data as the predicted value of 𝑥0. The way regression is implemented in by 
# having the calculation of the distances is being done for every sample in X_test.
# For every sample in X_Train, the distances are calculated, and they are sorted in ascending order. Then, 
# k nearest neighbors are found and y_predict is returned. This is done for every sample in X_Test. This 
# is why Knn-regression works.