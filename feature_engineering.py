# feature_engineering.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, title_map=None):
        """
        Constructor where you can optionally pass a custom title_map.
        """
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
        # No fitting is required for this transformer
        return self

    def transform(self, X):
        # Ensure 'Name' column exists before transforming
        if 'Name' not in X.columns:
            raise ValueError("Input data must contain a 'Name' column")

        X_ = X.copy()

        # Extract titles from 'Name' column
        X_['Title'] = X_['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Map extracted titles using the title_map
        X_['Title'] = X_['Title'].map(self.title_map).fillna('Other')

        # Family size feature
        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)

        return X_

    def set_params(self, **params):
        """
        Allows parameters like 'title_map' to be passed into the transformer during GridSearchCV or RandomizedSearchCV.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self