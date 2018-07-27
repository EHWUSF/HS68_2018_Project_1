import sklearn as sk
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss


class lr:
    '''
    implements the linear regression algorithm class
    '''

    def __init__(self, data_source, separator):
        self.beta = np.matrix(np.zeros(2))
        self.yhat = np.matrix(np.zeros(2))
        self.r2 = 0.0
        self.se = 0.0
        self.f = 0.0
        self.msr = 0.0
        self.mse = 0.0
        self.p = 0.0
        data_mat = np.genfromtxt(data_source, delimiter=separator)
        self.xarr = data_mat[:, :-1]
        self.yarr = data_mat[:, -1]
        self.ybar = np.mean(self.yarr)
        self.dfd = len(self.yarr) - 2  #df
        return


    def cov_custom(x, y):
        result = sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)
        return result


    def corr_custom(x, y):
        return lr.cov_custom(x, y) / (np.std(x, ddof=1) * np.std(y, ddof=1))


    def simple_regression(self):
        xmat = np.mat(self.xarr)
        ymat = np.mat(self.yarr).T
        xtx = xmat.T * xmat
        if np.linalg.det(xtx) == 0.0:
            print('Can not resolve the problem')
            return
        self.beta = np.linalg.solve(xtx, xmat.T * ymat)  # xtx.I * (xmat.T * ymat)
        self.yhat = (xmat * self.beta).flatten().A[0]
        return

    def r_square(self):
        y = np.mat(self.yarr)
        ybar = np.mean(y)
        self.r2 = np.sum((self.yhat - ybar) ** 2) / np.sum((y.A - ybar) ** 2)
        return


    def estimate_deviation(self):
        y = np.array(self.yarr)
        self.se = np.sqrt(np.sum((y - self.yhat) ** 2) / self.dfd)
        return


    def sig_test(self):
        ybar = np.mean(self.yarr)
        self.msr = np.sum((self.yhat - ybar) ** 2)
        self.mse = np.sum((self.yarr - self.yhat) ** 2) / self.dfd
        self.f = self.msr / self.mse
        self.p = ss.f.sf(self.f, 1, self.dfd)
        return

    def summary(self):
        self.simple_regression()
        corr_coe = lr.corr_custom(self.xarr[:, -1], self.yarr)
        self.r_square()
        self.estimate_deviation()
        self.sig_test()
        print('The Pearson\'s correlation coefficient: %.3f' % corr_coe)
        print('The Regression Coefficient: %s' % self.beta.flatten().A[0])
        print('R square: %.3f' % self.r2)
        print('The standard error of estimate: %.3f' % self.se)
        print('F-statistic:  %d on %s and %s DF,  p-value: %.3e' % (self.f, 1, self.dfd, self.p))


    def relationship_plots(df, x, y):
        # matplotlib
        # Plot the results
        sns.pairplot(df, x_vars=x, y_vars=y, height=7,
                     aspect=0.8, kind="reg")
        return plt.show()


def main(data_source_link):
    # Load the test csv dataset
    df = pd.read_csv(data_source_link,index_col=0)
    print df.shape
    df.head()
    x = raw_input('Which independent variable you want to plot and evaluate? (type the column name here)\n')
    y = raw_input('What is the dependent variable in this data frame? (type the column name here)\n')
    relationship_discover_plot = relationship_plots(df, x, y)
    summary_note = summary(data_source_link)
    print relationship_discover_plot
    print summary_note


if __name__ == '__main__':
    main()



