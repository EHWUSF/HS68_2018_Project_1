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

    def __init__(self, mean=0, std=1, data=None): ##default values input for mean and std
        """
        Actions:
        (0) Define parameters
        (1) Load data
        (2) Clean data by removing none numeric values("None", "Norm", and etc.)
        (3) Convert values from string to float

        Unit tests:
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
            with open("mydata.csv") as f:
                reader = csv.reader(f)  ##load data.csv
                next(reader)
                self.data = [r for r in reader]
                for i in range(len(self.data)):  ##remove None in data.csv
                    self.data[i] = [elem for elem in self.data[i] if elem.isnumeric()]
                self.data = np.array(self.data, dtype=np.float)  # convert to float
                print(self.data)

    ##
    def cal_outliers(self):
        """
        Description:
        Function for calculating values out of scope of two standard diviation in data and returning those values as outliers

        Parameters:
        (1) mean: mean of the list of list
        (2) std: standard diviation of the list
        """
        mean = sum(self.data) / (len(self.data))
        for i in range(0, len(self.data)):
            sum_diff = sum((self.data[i] - mean)) ** 2
            std = sum_diff ** 0.5
        outliers = [self.data[i] for i in range(0, len(self.data)) if abs(sum_diff) > 1.96 * std]
        return outliers

if __name__ == '__main__':
    import doctest
    doctest.testmod()

