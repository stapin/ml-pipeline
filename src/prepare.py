import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["prepare"]

df = pd.read_json('data/All_Beauty.jsonl', lines=True)

df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['full_text'] = df['title'] + " " + df['text']

X = df[['full_text']]
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=params['test_size'], 
    random_state=params['random_state'], 
    stratify=y 
)

os.makedirs("data/prepared", exist_ok=True)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data/prepared/train.csv", index=False)
test_df.to_csv("data/prepared/test.csv", index=False)