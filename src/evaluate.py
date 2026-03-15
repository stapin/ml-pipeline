import pandas as pd
import json
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

test_df = pd.read_csv("data/prepared/test.csv")
test_df['full_text'] = test_df['full_text'].fillna('')

X_test, y_test = test_df.drop('rating', axis=1), test_df['rating']

model = CatBoostRegressor()
model.load_model("models/model.cbm")

raw_predictions = model.predict(X_test)
final_predictions = np.clip(np.round(raw_predictions), 1, 5).astype(int)

rmse = root_mean_squared_error(y_test, final_predictions)

with open("metrics.json", "w") as f:
    json.dump({"rmse": rmse}, f)