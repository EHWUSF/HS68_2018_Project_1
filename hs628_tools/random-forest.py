import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing

### Function to wrap sklearn's randomforestclassifier and prediction model and
### return a tuple: predictions of test data, ranked scores of features.
def rf(data, features, target):
    features['train'] = np.random.uniform(0, 1, len(data)) <= 0.75
    train_features, test_features = features[features['train']==True], features[features['train']==False]
    train_target, test_target = target[target['train']==True], target[target['train']==False]

    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train_features, train_target)
    clf.predict(test_features)

    # Fix this indexing, does it even work?!
    predictions = test_features[clf.predict(test_features)]
    scores = list(zip(train_features, clf.feature_importances_))


    return predictions, scores



if __name__ == '__main__':

    pass