import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing

### Let the user decide between randomforest classifier or regressor. Return tuple of predictions and scores.
def rf(features, target, model=classifier):
    if model == 'classifier':
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
    elif model == 'regressor':
        pass

if __name__ == '__main__':

    pass