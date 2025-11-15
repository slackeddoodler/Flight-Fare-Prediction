import joblib

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

def prepare_df():

    df = pd.read_csv('dataset.csv')

    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['stops'] = df['stops'].replace({'zero': 0, 'one': 1, 'two_or_more': 2}).astype('int64')

    return df

def train_model(df):
    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(n_estimators=10)
    )

    train_dicts = df.drop(columns='price').to_dict(orient='records')
    y_train = np.log1p(df['price'].values)

    pipeline.fit(train_dicts, y_train)

    return pipeline

df = prepare_df()
model = train_model(df)
joblib.dump(model, "model.bin")