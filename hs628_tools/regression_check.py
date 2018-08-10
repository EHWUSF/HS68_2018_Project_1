# coding: utf-8

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
# %config IPCompleter.greedy=True
# %matplotlib inline

#   ****** Data Visualization to check Linear regression assumptions and calculating new regression metric ******

class RegCheck():
    """
        This Class is created for checking Linear Regression assumptions and Evaluating the Model.
        Some of the ideas and concepts for linear regression check are adapted from :
        https://github.com/dziganto/Data_Science_Fundamentals/blob/master/notebooks/Machine_Learning/
        Supervised_Learning/Regression/Linear_Regression/3_Linear_Regression_Assumptions_and_Evaluation.ipynb

        The following functions are defined in this class :

         func : check_linearity_residual_homoscedasticity_independence_plt checks to see if the
                data can fit to a linear model/non-linear model. Also check the homoscedasticity
                (check for the spread of data) of the residuals/error terms and also check for their
                independence. We diagnose these through a Residuals vs Predicted values plot.If there
                is no clear pattern and data symmetrically distributed it can be said to satisfy the
                assumptions.

         func : check_residuals_normality_plt checks to see the normality of the error terms through
                a density plot of the residuals. If the graph looks like gaussian normal curve it can
                be said that the residuals satisfy the assumption of normality.

         func : check_predictors_independence_plt checks for the independence of each predictor by
                plotting a heatmap of the correlations of each predictor with one and another. The
                heatmap would clearly indicate if there are multicolinear predictors.

         func : regression_metric builds a metric for evaluating the model. It is a harmonic mean of
                the R-Squared value and the F-Statistic taking into consideration the probability of
                significance of the F-statistic being significant than having an only intercept model.
    """

    def __init__(self):
        # Instantiating the parameters used in the methods below :PredictorColumnnames,R2,
        # F-stat,F-statpval,ResidualSumofSquares,RegressionMetric
        self.colnames = None
        self.rsquared = 0
        self.fstatistic = 0
        self.fstat_pval = 0
        self.resid_sumsq = 0
        self.regmetric = 0

    def check_linearity_residual_homoscedasticity_independence_plt(self, X, y, model):
        """
        Plots the data fitted to a linear model/non-linear model and helps user in checking the homoscedasticity
         of the residuals/error terms and also checks for their independence.This method has a predict method
         from scikit learn applied on the model to predict the values
            -----------
            Param:
                X: 2-D numpy array of the predictors
                y: 1-D numpy array of the response variables
                model : Linear regression model fitted to X(predictors) and y(response)
            -----------
            Return:
                Doesn't return anything, just shows the plot object built
        """

        # Plot of Residuals vs predicted values from the linear model
        fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.suptitle('Residual Plots')
        axes.plot(model.predict(X), y - model.predict(X), 'bo--')
        axes.axhline(y=0, color='k')
        axes.grid()
        axes.set_title('Linear/Non-Linear')
        axes.set_xlabel('predicted values')
        axes.set_ylabel('residuals')
        plt.show()

    def check_residuals_normality_plt(self, X, y, model):
        """
        Plots the data fitted to a linear model/non-linear model and helps user in checking
        the normality of the error terms through a density plot of the residuals
            -----------
            Param:
                X: 2-D numpy array of the predictors
                y: 1-D numpy array of the response variables
                model : Linear regression model fitted to X(predictors) and y(response)
            -----------
            Return:
                Doesn't return anything, just shows the plot object built
        """
        residuals = y - model.predict(X)

        # Density plot of the residuals
        sns.distplot(residuals)
        plt.xlabel("Residuals")
        plt.title('Residual Normality Check')
        plt.show()

    def check_predictors_independence_plt(self, X, y, colnames):
        """
        Plots the data fitted to a linear model/non-linear model and helps user in checking
        the independence of the predictors by plotting a heatmap of the correlations
            -----------
            Param:
                X: 2-D numpy array of the predictors
                y: 1-D numpy array of the response variables
                colnames : Column names of the predictors
            -----------
            Return:
                Doesn't return anything, just shows the plot object built
        """
        self.colnames = colnames
        corr_matrix = np.corrcoef(X, rowvar=False)

        # Heatmap of the correlation matrix
        sns.heatmap(corr_matrix, xticklabels=self.colnames, yticklabels=self.colnames, cmap='binary', annot=True)
        plt.suptitle('Heatmap of Correlations')
        plt.show()

    def regression_metric(self, X, y):
        """
        Calculates a new regression_metric for evaluating the model. R-Squared,F-Statistic,
        Residual Standard error are used from the summary of the model to calculate this metric.
        Intiuitively The RSE represents the average distance of the observed data from the model
        thus the lower the RSE the better fit and F-value expresses how much of the model has improved
        (compared to the mean) given the inaccuracy of the model and R-squared.R2 expresses how much
        of the total variation in the data can be explained by the mode(the regression line).

        Thus a harmonic between RSE/F-statistic and R2 is computed as a new metric.
        It ranges from 0 - 1. 0 being the worst and 1 being the highest or the best fit.
        Disclaimer : This is a original metric thought out from my knowledge of Linear regression metrics
        I have tried my best to think of all edge cases it might not fail, but is not foolproof.
            -----------
            Param:
                X: 2-D numpy array of the predictors
                y: 1-D numpy array of the response variables
            -----------
            Return:
                Returns a regression metric that defines the overall fit.
        """
        X = sm.add_constant(X)
        result = sm.OLS(y, X).fit()

        # Metrics required to calculate new reg metric extracted from the result object
        self.rsquared = result.rsquared
        self.fstatistic = result.fvalue
        self.fstat_pval = result.f_pvalue
        self.resid_sumsq = (result.mse_resid) ** 2

        if self.fstat_pval <= 0.05 or self.rsquared >= 0:
            self.regmetric = (1 / ((1 / self.rsquared) + ((self.resid_sumsq ** 0.5) / self.fstatistic)))
        else:
            self.regmetric = 0

        return self.regmetric


if __name__ == '__main__':

    # Taking the input file as a command line argument and checking the number of arguments : 2
    # (as script name is also considered as an argument)
    if len(sys.argv) == 2:
        fn = sys.argv[1]
    else:
        sys.exit('Filename not passed as an argument please check')

    # Checking if the file exists in the path if exists getting the absolute path
    if os.path.exists(fn):
        filepath = os.path.abspath(fn)
    else:
        raise ValueError("Input file name not found")

    # Loading the sample data to build and test the model
    data_load = np.genfromtxt(filepath, delimiter=",", dtype=float, names=True)
    # Converting the structured array into a normal numpy array
    data = data_load.view(np.float64).reshape(data_load.shape + (-1,))
    # Extract the Column names of the predictors
    names = list(data_load.dtype.names)
    predictor_names = names[:len(names) - 1]

    # Splitting the predictors and response into 2 seperate numpy arrays to build and fit the model
    datax = data[:, :-1]  # a 2-d array of all predictor variables
    datay = data[:, -1]  # a 1-d array of response variable

    # Fitting the LinearReg to the data
    lr = LinearRegression()
    lr.fit(datax, datay)

    # Instantiating a reg object from the RegCheck class
    reg = RegCheck()

    # Calling all the methods to run the diagnostic on the data
    reg.check_linearity_residual_homoscedasticity_independence_plt(datax, datay, lr)
    reg.check_residuals_normality_plt(datax, datay, lr)
    reg.check_predictors_independence_plt(datax, datay, predictor_names)
    reg_metric = reg.regression_metric(datax, datay)
    print("The new regression metric to evaluate the model is : " + str(reg_metric))
