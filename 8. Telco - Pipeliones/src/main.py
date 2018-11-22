from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
import numpy as np

from src.transformers import PandasSelector, Viewer
from src.preprocessing import DataCleaner, FillEmpty
from src.feature_engineering import PartnerAndDependent
from src import utilities

# Setup
desired_width = 80
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Load data
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.columns = [utilities.convert(x) for x in data.columns]
print("Data columns: ", data.columns)

model = GradientBoostingClassifier()

pipeline = make_pipeline(
    DataCleaner(),
    FillEmpty(),
    make_union(
        make_pipeline(
            PandasSelector(columns=['partner', 'dependents']),
            PartnerAndDependent(),
            utilities.one_hot_encoder()
        ),
        make_pipeline(
            PandasSelector(dtype=np.object),
            utilities.one_hot_encoder(),
        ),
    ),
    model
)

train_y = data['churn'].replace({'Yes': 1, 'No': 0})
train_x = data.drop('churn', axis=1).copy()

pipeline.fit(train_x, train_y)

print("Done fitting")
print("Scoring: ", pipeline.score(train_x, train_y))
