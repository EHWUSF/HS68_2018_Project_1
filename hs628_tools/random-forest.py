import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def rf(features, target, model='classifier', **kwargs):
    """Function to combine sklearn's randomforestclassifier and randomforestregressor methods

        Args:
            features (pandas dataframe or np array): pd dataframe or np array of feature or predictor variables
            target (pandas dataframe or np array): pd dataframe or np array of target or response variables
            model (str) : User chosen method, be it classifier or regressor. Default set to 'classifier'.

        Returns:
            list: list of feature importances
    """
    train_features, train_target, test_features, test_target = train_test_split(features, target, test_size=0.25, random_state=13)
    if model == 'classifier':
        my_defaults = {'n_estimators': 10, 'criterion': 'gini', 'max_features': 'auto', 'n_jobs': 1,
                       'random_state': 13, 'class_weight': 'balanced'}
        for k, v in my_defaults.iteritems():
            if k not in kwargs:
                kwargs[k] = v

        clf = RandomForestClassifier(**kwargs)
        clf.fit(train_features, train_target)
        clf.predict(test_features)
        predictions = clf.predict(test_target)
        scores = list(zip(train_features, clf.feature_importances_))

        return predictions, scores
    elif model == 'regressor':
        my_defaults = {'n_estimators': 20, 'max_features':'auto'}
        for k, v in my_defaults.iteritems():
            if k not in kwargs:
                kwargs[k] = v

        reg = RandomForestRegressor(**kwargs)
        reg.fit(train_features, train_target)
        reg.predict(test_features)
        predictions = reg.predict(test_target)
        scores = list(zip(train_features, reg.feature_importances_))

        return predictions, scores



### MAIN ###
if __name__ == '__main__':

    df = pd.read_csv('data.csv')

    X = df.drop('readmission_30d', axis=1)
    print type(X)
    X = X.drop('A1Cresult', axis=1)
    y = df['readmission_30d']

    train_features, train_target, test_features, test_target = train_test_split(X, y, test_size=0.25,
                                                                                random_state=13)
    clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto',
                                 n_jobs=1, random_state=13, class_weight='balanced')
    clf.fit(train_features, train_target)
    clf.predict(test_features)
    predictions = clf.predict(test_target)
    scores = list(zip(train_features, clf.feature_importances_))

    rfc = rf(X, y, model='classifier')
    rfr = rf(X, y, model='regressor')
