import numpy as np
import seaborn.apionly as sns
import matplotlib.pyplot as plt

Class Outliers:
    """
    Concept: 
    Returns list of data(list array) which fall outside of two standard deviations under the normal distribution curve.
    
    Parameters: 
    mean, standard deviation(std)
    
    Returns: 
    outliers    
    """
    def __init__(self, mean, std, outliers):
        self.mean = mean
        self.std = std
        self.outliers = outliers

    ## Function for calculating numbers out of scope of two standard diviation in dataset
    def outliers(self, dataset):
        outliers = [] #outliers is a list of data(array)
        mean = sum(dataset) / (len(dataset))
        sum_diff = sum((dataset[i] - mean))**2 for i in range(0, len(dataset)))/(len(dataset))
        std = sum_diff**0.5
        outliers = [tem[i] for i in range(0, len(dataset)) if abs(diff) > 1.96*std]
        print outliers

