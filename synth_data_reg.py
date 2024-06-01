import pandas as pd
import numpy as np

df = pd.read_csv("Dataset/ObesityDataSet.csv")

noise_levels = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

for noise_level in noise_levels:
    noisy_df = df.copy()

    mean_weight = noisy_df['Weight'].mean()

    for i in range(len(noisy_df)):
        random_noise = np.random.normal(0, noise_level * mean_weight)

        noisy_df.iloc[i, noisy_df.columns.get_loc('Weight')] += random_noise

    filename = f"Dataset/Dataset_weight_noisy_{int(noise_level*100)}percent.csv"
    noisy_df.to_csv(filename, index=False)