# -*- coding: utf-8 -*-
# Python version 3.6

import re
import numpy as np
import csv

from numpy import NaN


class Outliers:
    """
    Concept:
    Returns list of data(list of array) which fall outside of two standard deviations under the normal distribution curve.
    Parameters:
    mean, standard deviation(std)
    Returns:
    outliers
    *Note:
    This class is written for Python version 3.6.
    """

    def __init__(self):  # default values input for mean and std
        self.mean = None
        self.std = None
        self.data = None


    def calc_outliers(self, my_arr, outlier_limit=2):
        """
        Concept:
            Function for calculating samples out of scope of two standard deviation in data and returning those collection
            of samples as outliers
        Parameters:
            my_arr:
            outlier_limit: the number of standard diviations outside of which data points considered outliers
        Actions:
            Get quartile values for outlier limits: out of two standard deviation
        References:
            - Anomaly detection in Python at Lynda.com
              (https://www.lynda.com/Business-Intelligence-tutorials/Anomaly-detection-Python/475936/529731-4.html)
            - A Bayesian Anomaly Detection Framework for Python
              (pyISC: A Bayesian Anomaly Detection Framework for Python;
              Proceedings of the Thirtieth International Florida Artificial Intelligence Research Society Conference)
        """
        self.mean = my_arr.mean(axis=0)
        self.std = my_arr.std(axis=0)
        outliers = np.ones((my_arr.shape[0],), dtype=np.bool)
        for i in range(len(my_arr.shape)):  # added for detect outliers by each column
            col = my_arr[:, i]
            outliers[outliers] &= np.abs((col[outliers] - self.mean[i]) / self.std[i]) < outlier_limit
        return my_arr[outliers]


def read_testdata(): #'main' function should be out of 'Class'; should be other name to in it
    with open("mydata.csv") as f:
        # data.csv download link: https: // usf - mshi.slack.com / files / U6TC6KH0X / FBRM98U8G / data.csv
        reader = csv.reader(f)
        next(reader)
        data = [r for r in reader]
    conversion = np.array(data).astype("f4")
    return conversion


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    result = read_testdata()
    outlier_finder = Outliers()
    outlier_finder.calc_outliers(result)



