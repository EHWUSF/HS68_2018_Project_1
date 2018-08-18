# -*- coding: utf-8 -*-
# Python version 3.6

import numpy as np
import csv


class Outliers:
    """
    * Note:
      This class is written for Python version 3.6.
    -------------------------------------------------------------------------------------------------------------------
    Concept:
    Filter out samples which fall outside of two standard deviations under the normal distribution curve.
    Returns:
    List of samples collected as outliers
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.data = None


    def calc_outliers(self, my_arr, outlier_limit=2):
        """
        - The purpose of this function is to calculate samples out of scope of two standard deviation in data, which is
          the range of outliers, and to return those collected samples as outliers
        Parameters:
            my_arr: the list of numpy arrays which is returned from outside function of "read_testdata".
            outlier_limit: the number of standard deviations outside of which data points considered outliers
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
        for i in range(len(my_arr.shape)):
            col = my_arr[:, i]
            outliers[outliers] &= np.abs((col[outliers] - self.mean[i]) / self.std[i]) < outlier_limit
        return my_arr[outliers]


def read_testdata():
    """
    - The purpose of this function is to read data set.
    - The function is added code lines converting data to numpy array, and data type from 'string' to 'float'.
    ----------------------------------------------------------------------------------------------------------------
    * Note that "data.csv" used in this project is the version the "A1Cresult" column is removed from the original
                "data.csv" file. The file can be downloaded through the link,
                "https: // usf - mshi.slack.com / files / U6TC6KH0X / FBRM98U8G / data.csv"
    """
    with open("data.csv") as f:
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



