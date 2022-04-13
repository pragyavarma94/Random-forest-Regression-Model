from pypmml import Model 
import pandas as pd 

train_df = pd.read_csv('boston_train.csv')
print(train_df)

test_df = pd.read_csv('boston_test.csv')
print(test_df)

train_df = train_df.apply(pd.to_numeric, errors='ignore')
test_df = test_df.apply(pd.to_numeric, errors='ignore')

features = train_df.columns.drop('MEDV')

model = Model.load('cpy_of_original.pmml')
predictions = model.predict(test_df[features])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   print(predictions)

