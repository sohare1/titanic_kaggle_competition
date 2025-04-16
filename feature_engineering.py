# feature_engineering.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, title_map=None):
        # scikit-learn needs all params in __init__, even defaults
        self.title_map = title_map or {
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Ms": "Miss",
            "Mlle": "Miss",
            "Mme": "Mrs",
            "Master": "Master",
            "Dr": "Officer",
            "Rev": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Capt": "Officer",
            "Sir": "Royalty",
            "Lady": "Royalty",
            "Don": "Royalty",
            "Jonkheer": "Royalty",
            "Countess": "Royalty"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # Title Feature
        X_['Title'] = X_['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        X_['Title'] = X_['Title'].map(self.title_map).fillna('Other')

        # Family Features
        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)

        return X_
