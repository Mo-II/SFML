import pandas as pd

# Read the CSV files
df1 = pd.read_csv('test.csv')
df2 = pd.read_csv('train.csv')

# Merge the DataFrames (example: concatenate by rows)
merged_df = pd.concat([df2, df1])

# Write the merged data to a new CSV file
merged_df.to_csv('Dataset.csv', index=False)