import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def rf(features, target, model='classifier'):
    train_features, train_target, test_features, test_target = train_test_split(features, target, test_size=0.25, random_state=13)
    if model == 'classifier':
        clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto',
                                     n_jobs=1, random_state=13, class_weight='balanced')
        clf.fit(train_features, train_target)
        clf.predict(test_features)
        predictions = clf.predict(test_target)
        scores = list(zip(train_features, clf.feature_importances_))

        return predictions, scores
    elif model == 'regressor':
        reg = RandomForestRegressor(n_estimators=20, max_features='auto')
        reg.fit(train_features, train_target)
        reg.predict(test_features)
        predictions = reg.predict(test_target)
        scores = list(zip(train_features, reg.feature_importances_))

        return predictions, scores



### MAIN ###
if __name__ == '__main__':

    pass