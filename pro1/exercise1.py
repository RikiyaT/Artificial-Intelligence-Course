import numpy as np
class KNearestNeighborsClassifier():
    #passing variable "k"
    def __init__(self, k=5):
        self.k = k
        self.dist_metric = self.euclidean
    #passing variable "init"
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # distances calculates the distance between x and x_Train
    #The training dataset classes are sorted by distance to the data point
    #The first k classes are kept and stored in the neighbors list 
    # Now we simply map the list of nearest neighbors to our most_common function, 
    # returning a list of predictions for each point passed in X_test.
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(self.most_common, neighbors))
    
    def most_common(self, lst):
        '''Returns the most common element in a list'''
        return max(set(lst), key=lst.count)

    def euclidean(self, point, data):
        '''Euclidean distance between a point  & data'''
        return np.sqrt(np.sum((point - data)**2, axis=1))

# check this is a main file
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris_dataset = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data,
                                                        iris_dataset.target,
                                                        random_state=0)
    knn = KNearestNeighborsClassifier(3)
    knn.fit(X_train, Y_train)
    X_test_predict = knn.predict(X_test)
    accuracy = np.sum(Y_test == X_test_predict) / Y_test.shape[0]
    assert(accuracy > 0.7)
    print('acc:', accuracy)


