import pandas as pd
import yaml
import os
from catboost import CatBoostClassifier, Pool

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

train_df = pd.read_csv("data/prepared/train.csv")
test_df = pd.read_csv("data/prepared/test.csv")

train_df['full_text'] = train_df['full_text'].fillna('')
test_df['full_text'] = test_df['full_text'].fillna('')

X_train, y_train = train_df.drop('rating', axis=1), train_df['rating']
X_test, y_test = test_df.drop('rating', axis=1), test_df['rating']

text_cols = ['full_text']

train_pool = Pool(data=X_train, label=y_train, text_features=text_cols)
test_pool = Pool(data=X_test, label=y_test, text_features=text_cols)

model = CatBoostClassifier(
    iterations=params['iterations'],
    learning_rate=params['learning_rate'],
    eval_metric='RMSE',
    task_type='CPU',              
    early_stopping_rounds=params['early_stopping_rounds'],
    verbose=100                  
)

model.fit(train_pool, eval_set=test_pool)

os.makedirs("models", exist_ok=True)
model.save_model("models/model.cbm")