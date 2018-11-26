import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.transformers import PandasSelector, Viewer
from src.preprocessing import DataCleaner, FillEmpty
from src.feature_engineering import PartnerAndDependent, ChangesChange
from src import utilities

# Setup
desired_width = 80
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(threshold=100)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Load data
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.columns = [utilities.convert(x) for x in data.columns]
print("Data columns: ", data.columns)

data_y = data['churn'].replace({'Yes': 1, 'No': 0})
data_x = data.drop('churn', axis=1).copy()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

preprocessing = make_pipeline(
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
        make_pipeline(
            PandasSelector(columns=['monthly_charges', 'tenure', 'total_charges']),
            StandardScaler(),
        ),
        make_pipeline(
            PandasSelector(columns=['monthly_charges', 'tenure', 'total_charges']),
            ChangesChange(),
            StandardScaler()
        )
    ),
    # Viewer(),
)

model = lgb.LGBMClassifier()

modelling = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', model),
    ]
)

modelling.fit(train_x, train_y)

print("Done fitting")
print("Scoring: ", modelling.score(train_x, train_y))
