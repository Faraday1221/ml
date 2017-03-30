
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class Adaline(object):
    "2-D Adaline classifier"
    def __init__(self,learning_rate=0.1,epoch=10):
        self.learning_rate = learning_rate
        self._epoch = epoch

    def fit(self,X,y):
        # initialize the weights to zero (with intercept)
        w = np.zeros(X.shape[1])
        b = 1
        n = self.learning_rate

        # perhaps we just need more learning
        for _ in range(self._epoch):
            # for each training example perform the following
            for i in range(X.shape[0]):
                # compute the net output
                z = b + X.iloc[i,0]*w[0] + X.iloc[i,1]*w[1]

                # update the weights
                b += n*(y.iloc[i] - z)
                w[0] += n*(y.iloc[i] - z)*X.iloc[i,0]
                w[1] += n*(y.iloc[i] - z)*X.iloc[i,1]

        self.w = w
        self.b = b

    def predict(self,X):
        pred = []
        # for each training example x^(i) perform the following:
        for i in range(X.shape[0]):
            # 1. compute the output value y
            z = self.b + X.iloc[i,0]*self.w[0] + X.iloc[i,1]*self.w[1]

            if z >= 0.0:
                y_pred = 1
            else:
                y_pred = -1
            pred.append(y_pred)

        return np.array(pred)

    def score(self,X,y):
        """returns the percent correct classifications"""
        correct = (self.predict(X) == y).sum()
        total = y.shape[0]
        return float(correct)/total




if __name__ == '__main__':
    # import our dataset that is guaranteed to converge
    df = pd.read_csv("../perceptron/toy_data2.csv")
    X = df[['x1','x2']]
    y = df['y']


    # a full example using the class above
    clf = Adaline(learning_rate=0.005, epoch=50)
    clf.fit(X,y)
    print "Accuracy Score {0} %".format(clf.score(X,y)*100)

    # plot of error against the learning rate
    eta = []
    for n in [.0001,.0003,.001,.003,.01,.03]:
        clf = Adaline(learning_rate = n, epoch=100)
        clf.fit(X,y)
        eta.append(clf.score(X,y))

    plt.plot(eta, 'ko-')
    plt.show()
