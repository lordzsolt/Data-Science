from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, x, y):
        return self

    @staticmethod
    def transform(x: pd.DataFrame) -> pd.DataFrame:
        x['senior_citizen'] = x['senior_citizen'] == 1
        return x.replace({'Yes': True, 'No': False})


class FillEmpty(BaseEstimator, TransformerMixin):
    def fit(self, x, y):
        return self

    @staticmethod
    def transform(x: pd.DataFrame) -> pd.DataFrame:
        total_charges = pd.to_numeric(x['total_charges'], errors='coerce')
        empty = total_charges[total_charges.isna()]
        total_charges[empty.index.values] = 0
        x['total_charges'] = total_charges

        return x
