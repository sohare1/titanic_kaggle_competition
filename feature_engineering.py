# feature_engineering.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, title_map=None):
        # Default title map if none is passed
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
        # Fit does nothing for this transformer
        return self

    def transform(self, X):
        X_ = X.copy()

        # Check if 'Name' column exists
        if 'Name' not in X_.columns:
            raise ValueError("Input data must contain a 'Name' column")

        # Title Feature extraction
        X_['Title'] = X_['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Debugging: Check if titles were extracted correctly
        print("Titles extracted:", X_['Title'].unique())

        # Mapping titles to groups
        X_['Title'] = X_['Title'].map(self.title_map).fillna('Other')

        # Check if mapping worked
        print("Mapped Titles:", X_['Title'].unique())

        # Family Features (Family size and IsAlone)
        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)

        return X_
