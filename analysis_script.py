import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load
trades = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')
sent = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')

# Clean/parse
trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
trades['date'] = trades['Timestamp IST'].dt.date
sent['date'] = pd.to_datetime(sent['date']).dt.date
sent = sent[['date','classification','value']].rename(columns={'classification':'sentiment','value':'sentiment_value'})

# Merge
df = trades.merge(sent, on='date', how='left')
print("Unmatched dates:", df['sentiment'].isna().sum(), "of", len(df))

df = df.dropna(subset=['sentiment']).reset_index(drop=True)
df.to_pickle('/home/claude/analysis/merged.pkl')
print(df.shape)
print(df['sentiment'].value_counts())
