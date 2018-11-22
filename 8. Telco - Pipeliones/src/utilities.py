import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer


def convert(name: str) -> str:
    """
    Convert upper and lower camel case to snake case
    :param name: String to convert
    :return: Converted string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_records(df):
    return df.to_dict(orient='records')


def one_hot_encoder():
    return make_pipeline(
        FunctionTransformer(to_records, validate=False),
        DictVectorizer()
    )
