
# coding: utf-8

# In[1]:


import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#%config IPCompleter.greedy=True


# ************************ Part 1 - Data Visualization to check Linear regression assumptions ******************

# In[ ]:


class reg_check:
    
    def __init__(self, X, y, model):
        self.data = X
        self.target = y
        self.model = model
        ## degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1   
        ## degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1  


# In[ ]:


    def check_linearity_plt(X,y,linear):

        # Plot of Residuals vs individual independent variables(predictors).
        for i in len(X):
            Y_pred = linear.predict(X)
            residual = y-Y_pred
            plt.scatter(,residual)
            plt.xlabel("X1 - a predictor")
            plt.ylabel("residual")
            plt.show()

            fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
            fig.suptitle('[Residual Plots]')

            axes[i].axhline(y=0, color='k')
            axes[i].grid()
            axes[i].set_title('Linear')
            axes[i].set_xlabel('predicted values')
            axes[i].set_ylabel('residuals')


        # Plot of Residuals vs predicted values
        fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.suptitle('[Residual Plots]')
        fig.set_size_inches(12,5)
        axes[0].plot(linear.predict(X), y-linear.predict(X), 'bo')
        axes[0].axhline(y=0, color='k')
        axes[0].grid()
        axes[0].set_title('Linear')
        axes[0].set_xlabel('predicted values')
        axes[0].set_ylabel('residuals')





# In[ ]:


    def check_predictors_independence_plt(X,y,linear):



# In[ ]:


    def check_residuals_independence_plt(X,y,linear):



# In[ ]:


    def check_residuals_normality_plt(X,y,linear):



# In[ ]:


    def check_residuals_homoscedasticity_plt(X,y,linear):


# In[ ]:


    def check_regression_assumptions(X,y):
        # Call all the above 5 functions


# In[ ]:


# Call the check_assumptions function to print all the plots and check for linear regression assumptions


# ****************** Part 2 - Calculate new metric to report the model ****************
