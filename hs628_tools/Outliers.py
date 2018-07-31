import pylab
import numpy as np


class Outliers:
    """
    Concept:
    Returns list of data(list array) which fall outside of two standard deviations under the normal distribution curve.

    Parameters:
    mean, standard deviation(std)

    Returns:
    outliers

    *Note:
    Assumed the data set('data.csv') is loaded
    """

    def __init__(self, mean=0, std=1, data=None):  # default values input for mean and std
        """
        Actions:
        (0) Define parameters
        (1) Load data
        (2) Clean data by removing none numeric values("None", "Norm", and etc.)
        (3) Convert values from string to float

        Unit tests to confirm data cleaning:
          >>> "6".isnumeric()
          True
          >>> "None".isnumeric()
          False
          >>> "Norm".isnumeric()
          False
        """
        self.mean = mean
        self.std = std
        self.data = data

        if data is None:
            for i in range(len(self.data)):  # remove None in data.csv
                self.data[i] = [elem for elem in self.data[i] if elem.isnumeric()]
            self.data = np.array(self.data, dtype=np.float)  # convert to float
            print(self.data)
    ##
    def cal_outliers(my_arr, out_lim):
        """
        Concept:
        Function for calculating values out of scope of two standard deviation in data and returning those values as
        outliers

        Actions:
        Get quartile values for outlier limits: out of two standard deviation

        References:
        - Anomaly detection in Python at Lynda.com
          (https://www.lynda.com/Business-Intelligence-tutorials/Anomaly-detection-Python/475936/529731-4.html)
        - A Bayesian Anomaly Detection Framework for Python
          (pyISC: A Bayesian Anomaly Detection Framework for Python;
                 Proceedings of the Thirtieth International Florida Artificial Intelligence Research Society Conference)
        """
        my_arr = np.random.randn(20, 10)  # data array
        out_lim = np.selp.std*1.96  # set outlier limit
        outliers = np.ones((my_arr.shape[0],), dtype=np.bool)
        self.mean = np.self.mean(my_arr, axis=0)
        self.std = np.self.std(my_arr, axis=0, drop=1)
        for i in range(my_arr.shape[1]):  # added for detect outliers by each column
            col = my_arr[:, i]
            outliers[outliers] &= np.abs((col[outliers] - self.mean[i]) / self.std[i]) < out_lim
        return my_arr[outliers]


if __name__ == '__main__':
    import doctest
    doctest.testmod()

