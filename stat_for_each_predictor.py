import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import scipy.stats as ss


class lr():
    """
    implements the linear regression algorithm class
    include some stat summary of linear regression, such as R square, correlation coefficient, and standard error.
    Finally, it can help to evaluate a plot of the model.
    """

    def __init__(self, data_source, nd_x):
        """
        Read the predictor you want to test and name the index in nd_x, it can be the 1st, 2nd or 3rd of X feature.
        But only one predictor per test.
        """
        self.beta = np.matrix(np.zeros(2))
        self.yhat = np.matrix(np.zeros(2))
        self.r2 = 0.0
        self.se = 0.0
        data_mat = np.genfromtxt(data_source, delimiter=',', skip_header=1)
        data_mat = data_mat[:, [0, nd_x]]
        self.xarray = data_mat[:, :-1]
        self.yarray = data_mat[:, -1]
        self.ybar = np.mean(self.yarray)
        self.dfd = len(self.yarray) - 2  #df of y


    def cov_custom(self):
        x = self.xarray
        y = self.yarray
        self.cov = sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)
        return


    def corr_custom(self):
        x = self.xarray
        y = self.yarray
        self.corr = self.cov / (np.std(x, ddof=1) * np.std(y, ddof=1))
        return


    def simple_regression(self):
        xmat = np.mat(self.xarray)
        ymat = np.mat(self.yarray).T
        xtx = xmat.T * xmat
        if np.linalg.det(xtx) == 0.0:
            print('This matrix is singular, cannot do inverse')
            return
        self.beta = np.linalg.solve(xtx, xmat.T * ymat) # or xtx.I * (xmat.T * ymat) in formula
        self.yhat = (xmat * self.beta).flatten().A[0]
        return

    def r_square(self):
        y = np.mat(self.yarray)
        ybar = np.mean(y)
        ssreg = np.sum((self.yhat - ybar) ** 2)
        sstot = np.sum((y.A - ybar) ** 2)
        self.r2 = ssreg / sstot
        return


    def estimate_deviation(self):
        y = np.array(self.yarray)
        self.se = np.sqrt(np.sum((y - self.yhat) ** 2) / self.dfd)
        return


    def summary(self):
        self.simple_regression()
        self.r_square()
        self.estimate_deviation()
        print('The Regression Coefficient: %s' % self.beta.flatten().A[0])
        print('R square: %.3f' % self.r2)
        print('The standard error of estimate: %.3f' % self.se)



if __name__ == '__main__':
    # use advertising.csv as test data set:
    # recall class:
    lr_test_data = lr(data_source='http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', nd_x=2)
    summary_note = lr_test_data.summary()
    print summary_note

