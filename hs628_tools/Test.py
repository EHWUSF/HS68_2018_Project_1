import numpy as np
from sklearn.model_selection import train_test_split


class DataSplit:
    """
    Create class data_split to predict a perfect train and test data split ratio.

    defines method to accomplish this task.

    returns a split ratio.

    Identifies Missing values as Nan for any missing value

    Returns:  train and test dataframe with uniform division
    of NAN between train and test data per the data split ratio.

    """


    def test_train(self):
        """

        Suggests train and test ratio

        set calculation standard for specific datalengths, this is based on the general rule of thumb
        ratio increases as the length of the data increases, so to ensure we have  enough data to train and test
        for the smaller dataset

        prints out the train_percentage for user's convenience

        Returns: new train and test set with uniform distribution of NAN values as tuple

        """

        row_len = len(datafile)
        if len(datafile) <= 500:
            train_percent = 60
        elif len(datafile) > 500 and row_len <= 10000:
            train_percent = 70
        elif len(datafile) > 10000 and row_len <= 100000:
            train_percent = 75
        else:
            train_percent = 80
        print 'Train percentage for this data set:', train_percent

        # yields rows that do not contain nan data. This is done to accumulate nan data in
        #seperate array and later, use the array to uniformly distribute among test and train data
        non_nandata = datafile[~np.isnan(datafile).any(axis=1)]


        ### array that contains rows with corresponding NAN values- to be used later to plug in
        #test and train set unifromly
        nan_data = datafile[np.isnan(datafile).any(axis=1)]

        ##split  Nan data and non_nandata using the predeterminded split ratio
        train_nan_data = nan_data[:int(train_percent* len(nan_data)/100), :]
        test_nan_data = nan_data[int(train_percent* len(nan_data)/100):, :]


        ##split non Nan data to train and test sets
        train_non_nan = non_nandata[:int(train_percent * len(non_nandata)/100), :]
        test_non_nan = non_nandata[int(train_percent * len(non_nandata)/100):, :]

        #Add train_non_nan and train_nan_data to create a final train data array
        final_train = np.concatenate([train_non_nan, train_nan_data])

        #### sum of  test_non_nan and test_nan_data
        final_test = np.concatenate([test_non_nan, test_nan_data])

        return "new_train_data :", final_train,\
               "new_test_data:", final_test


if __name__ == '__main__':
    data = "heart.csv"
    #'heart csv' can be downloaded from : http://www.cs.usfca.edu/~pfrancislyon/data/heart.csv
    datafile = np.genfromtxt(data, delimiter=',', skip_header=1)
    np.isnan(datafile).sum()  # counts the number of "NANs" in data
    new_test_train = DataSplit().test_train()
    print new_test_train











