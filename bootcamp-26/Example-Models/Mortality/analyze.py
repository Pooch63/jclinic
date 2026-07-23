import pandas as pd

df = pd.read_csv('hackathon_data_clean.csv')
print(df.columns.tolist())

print(df['Death'].value_counts())