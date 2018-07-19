import numpy as np
import csv
import matplotlib.pyplot as plt

##Class Outliers:
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

with open("data.csv") as f:
     reader = csv.reader(f)
     next(reader)
     data = [r for r in reader]

## Function for calculating numbers out of scope of two standard diviation in dataset
def outliers(self, data):
    outliers = [] #outliers is a list of data(array)
    mean = sum(data) / (len(data))
    for i in range(0, len(data)):
        sum_diff = sum((data[i] - mean))**2
        std = sum_diff**0.5
    outliers = [data[i] for i in range(0, len(data)) if abs(diff) > 1.96*std]
    print(outliers)

