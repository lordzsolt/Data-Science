from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class PartnerAndDependent(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        return self

    @staticmethod
    def transform(x: pd.DataFrame) -> pd.DataFrame:
        column = 'partner_and_dependents'
        x.loc[x['partner'] & x['dependents'], column] = 'Both'
        x.loc[~(x['partner']) & x['dependents'], column] = 'Just dependents'
        x.loc[x['partner'] & ~(x['dependents']), column] = 'Just partner'
        x.loc[~(x['partner'] | x['dependents']), column] = 'Neither'

        return x


class ChangesChange(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        return self

    @staticmethod
    def transform(x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(x['total_charges'] - x['monthly_charges'] * x['tenure'])
