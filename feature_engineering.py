import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Extracts Title, FamilySize, IsAlone from Titanic dataset."""

    def __init__(self):
        self.title_map = {
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

        # Title Extraction & Mapping
        X_['Title'] = X_['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        X_['Title'] = X_['Title'].map(self.title_map).fillna('Other')

        # Family Size Features
        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)

        return X_
