
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

def perceptron_data(rows=100,m=1,b=0):
    """Dataset that ensures convergence of the perceptron algorithm, i.e. linearly seperable classification
    Inputs:
        rows determines the number of observations in the dataset
        m corresponds to slope, intercept is defaulted to 0
    Returns:
        DataFrame (rows,3) with target guaranteed to converge using the
        perceptron algorithm, x1 and x2 between [0:1] and y [-1,1]
    """
    data = np.random.random(size=(2,rows))
    target = np.where(data[1,:]/data[0,:] > m+b,1,-1)

    df = pd.DataFrame(data=np.concatenate((data,[target]),axis=0).T,\
            columns=['x1','x2','y'])
    return df

def plot_perceptron_data(df):
    "plots the perceptron dataset"
    sns.lmplot( 'x1','x2',
                data=df,
                fit_reg=False,
                hue='y')
    sns.plt.title('Perceptron Toy Data')
    sns.plt.xlabel('x1')
    sns.plt.ylabel('x2')
    sns.plt.show()

if __name__ == '__main__':
    df = perceptron_data()
    # df.to_csv('perceptron_data.csv',index=False)

    plot_perceptron_data(df)
