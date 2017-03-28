
import numpy as np
import pandas as pd

from make_data import perceptron_data

# perceptron learning algorithm:
# initialize the weights to zero or small random numbers

# for each training example x^(i) perform the following:
# 1. compute the output value y
# 2. update the weights


class Perceptron(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def fit(self,X,y):
        """X expects a dataframe of x1 and x2
           target expects a series of y [-1,1]
           learning_rate must be > 0 and <= 1"""
        # initialize the weights to zero
        w = np.zeros(X.shape[1])
        n = self.learning_rate

        # for each training example x^(i) perform the following:
        for i in range(X.shape[0]):
            # 1. compute the output value y
            z = X.iloc[i,0]*w[0] + X.iloc[i,1]*w[1]

            if z >= 0:
                y_pred = 1
            else:
                y_pred = -1

            # 2. update the weights
            w[0] = w[0] + n*(y.iloc[i] - y_pred)*X.iloc[i,0]
            w[1] = w[1] + n*(y.iloc[i] - y_pred)*X.iloc[i,1]
        self.w = w

    def predict(self,X):
        """given a 2-D dataframe of x1 and x2 predict the output y [-1,1]
           returns a list of predicted classes [-1,1]"""
        pred = []
        # for each training example x^(i) perform the following:
        for i in range(X.shape[0]):
            # 1. compute the output value y
            z = X.iloc[i,0]*self.w[0] + X.iloc[i,1]*self.w[1]

            if z >= 0:
                y_pred = 1
            else:
                y_pred = -1
            pred.append(y_pred)

        return pred

    def score(self,X,y):
        """returns the percent correct classifications"""
        correct = (self.predict(X) == y).sum()
        total = y.shape[0]
        return "{0:.2f} % Accuracy".format(100.0*correct/total)




if __name__ == '__main__':
    # import our dataset that is guaranteed to converge
    df = perceptron_data(rows=100)
    X = df[['x1','x2']]
    y = df['y']

    # a full example using the class above
    clf = Perceptron(learning_rate=0.5)
    clf.fit(X,y)

    print clf.score(X,y)
