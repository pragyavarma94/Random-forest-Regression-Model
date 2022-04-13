import pyodbc
import pandas as pd
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.ensemble import RandomForestRegressor
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.preprocessing import CutTransformer
from sklearn.impute import SimpleImputer
import os
import time
import getpass as op
from teradataml.context.context import *
from IPython.display import display
import numpy as np

display.print_sqlmr_query = False

train_df = pd.read_csv('boston_train.csv')
print(train_df)

test_df = pd.read_csv('boston_test.csv')
print(test_df)

train_df = train_df.apply(pd.to_numeric, errors='ignore')
test_df = test_df.apply(pd.to_numeric, errors='ignore')

features = train_df.columns.drop('MEDV')
target = 'MEDV'

ct = CutTransformer(bins=np.linspace(1,10,10), right=True, labels=False)
ct.transform(train_df['RM'])

mapper = DataFrameMapper([
  (['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], StandardScaler()),
  (['RM'], [SimpleImputer(), ct])
  ])

pipeline = PMMLPipeline([
    ("mapping", mapper),
    ("rfc", RandomForestRegressor(n_estimators = 100, random_state = 0))
])

pipeline.fit(train_df[features], train_df[[target]].values.ravel())
sklearn2pmml(pipeline, "model.pmml", with_repr = True)

#predict using test data
#........................
from pypmml import Model
model = Model.load('model.pmml')
predictions = model.predict(test_df[features])
#print(predictions)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   print(predictions)