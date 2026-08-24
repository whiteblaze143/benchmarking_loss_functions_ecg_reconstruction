import pandas as pd
df = pd.read_csv("data/ptb_xl/ptbxl_database.csv")
print(df[['age', 'sex', 'height', 'weight']].isnull().sum())
print(df[['age', 'sex', 'height', 'weight']].head())
