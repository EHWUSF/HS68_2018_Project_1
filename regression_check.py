# coding: utf-8


import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# %config IPCompleter.greedy=True


# ************************ Part 1 - Data Visualization to check Linear regression assumptions ******************

class RegCheck():
    def __init__(self):
        self.data = None
        self.target = None
        self.model = None
        ## degrees of freedom population dep. variable variance
        self._dft = None
        ## degrees of freedom population error variance
        self._dfe = None

    def check_linearity_residual_homoscedasticity_independence_plt(self, X, y, linear):
        # Plot of Residuals vs predicted values
        fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.suptitle('[Residual Plots]')
        fig.set_size_inches(12, 5)
        axes.plot(linear.predict(X), y - linear.predict(X), 'bo--')
        axes.axhline(y=0, color='k')
        axes.grid()
        axes.set_title('Linear/Non-Linear')
        axes.set_xlabel('predicted values')
        axes.set_ylabel('residuals')
        plt.show()

    def check_residuals_normality_plt(self, X, y, linear):
        residuals = y - linear.predict(X)
        sns.distplot(residuals)
        plt.xlabel("Residuals")
        plt.title('Residual Normality Check')
        plt.show()

    def check_predictors_independence_plt(self, X, y, linear, names):
        data_df = pd.DataFrame(X, columns=names)
        sns.heatmap(data_df.corr(), cmap='binary', annot=True)
        plt.suptitle('Heatmap of Correlations')
        plt.show()


if __name__ == '__main__':
    # Call the check_assumptions function to print all the plots and check for linear regression assumptions
    data_load = np.genfromtxt('data.csv', delimiter=",", dtype=float, names=True)
    data = data_load.view(np.float64).reshape(data_load.shape + (-1,))
    names = list(data_load.dtype.names)
    predictor_names = names[:len(names) - 1]

    datax = data[:, :-1]  # a 2-d array of all predictor variables
    datay = data[:, -1]  # a 1-d array of response variable

    lr = LinearRegression()
    lr.fit(datax, datay)

    reg = RegCheck()

    # reg.check_linearity_plt(datax,datay,lr)
    # reg.check_linearity_residual_homoscedasticity_independence_plt(datax,datay,lr)
    # reg.check_predictors_independence_plt(datax,datay,lr,predictor_names)


# ****************** Part 2 - Calculate new metric to report the model ****************
