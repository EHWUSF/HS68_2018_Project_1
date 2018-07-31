import numpy as np
import csv

class Outliers:
    """
    Concept:
    Returns list of data(list array) which fall outside of two standard deviations under the normal distribution curve.

    Parameters:
    mean, standard deviation(std)

    Returns:
    outliers
    """

    def __init__(self, mean=0, std=1, mydata=None): ##default values input for mean and std
        """
        Actions:
        (0) Define parameters
        (1) Load data
        (2) Clean data by removing none numeric values("None", "Norm", and etc.)
        (3) Convert values from string to float

        Unit tests to confirm data cleaning
          >>> "6".isnumeric()
          True
          >>> "None".isnumeric()
          False
          >>> "Norm".isnumeric()
          False
        """
        self.mean = mean
        self.std = std
        self.mydata = mydata

        if data is None:
            for i in range(len(self.mydata)):  ##remove None in data.csv
                self.data[i] = [elem for elem in self.mydata[i] if elem.isnumeric()]
            self.mydata = np.array(self.mydata, dtype=np.float)  # convert to float
            print(self.mydata)
    ##
    def cal_outliers(self):
        """
        Description:
        Function for calculating values out of scope of two standard diviation in data and returning those values as outliers

        Parameters:
        (1) mean: mean of the list of list
        (2) std: standard diviation of the list
        """
        self.mean = sum(self.mydata) / (len(self.mydata))
        for i in range(0, len(self.mydata)):
            sum_diff = sum((self.mydata[i] - self.mean)) ** 2
            self.std = sum_diff ** 0.5
        outliers = [self.mydata[i] for i in range(0, len(self.mydata)) if abs(sum_diff) > 1.96 * self.std]
        return outliers

if __name__ == '__main__':
    import doctest
    doctest.testmod()

