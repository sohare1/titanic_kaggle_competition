import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to extract features like Title, FamilySize, and IsAlone from Titanic data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Title'] = X_['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        X_['Title'] = X_['Title'].replace([
            'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
        ], 'Rare')
        X_['Title'] = X_['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)

        return X_
