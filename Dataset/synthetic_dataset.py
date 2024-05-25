import numpy as np
from scipy.stats import norm, binom, poisson, uniform
import pandas as pd
from pre_processing import make_categorical


def create_synthetic_dataset(n_samples):
    df = pd.read_csv('ObesityDataSet.csv')

    #print("Original Gender Distribution:")
    #print(df['Gender'].value_counts())

    gender_imbalanced = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.2, 0.8])

    # Extract means and standard deviations from the original dataset
    synthetic_data = {}
    # Filtering data for males and females
    male_data = df[df['Gender'] == 'Male']
    female_data = df[df['Gender'] == 'Female']


    for column in df.columns:
        if column != 'Gender':

            #No integers in dataset right now ?
            """
            if df[column].dtype in [np.int64, np.int32]:  # Integer features
                # Data for males
                male_min_val = male_data[column].min()
                male_max_val = male_data[column].max()

                # Data for females
                female_min_val = female_data[column].min()
                female_max_val = female_data[column].max()

                # Generating synthetic data for males
                male_synthetic = np.random.randint(low=male_min_val, high=male_max_val + 1, size=int(0.2 * n_samples))

                # Generating synthetic data for females
                female_synthetic = np.random.randint(low=female_min_val, high=female_max_val + 1, size=int(0.8 * n_samples))

                # Concatenating synthetic data
                synthetic_data[column] = np.concatenate((male_synthetic, female_synthetic))
            """
            if df[column].dtype in [np.float64, np.float32]:  # Continuous features
                # For males
                male_mean = male_data[column].mean()
                male_std = male_data[column].std()
                male_synthetic = np.abs(np.random.normal(loc=male_mean, scale=male_std, size=int(0.2 * n_samples)))
                # For females
                female_mean = female_data[column].mean()
                female_std = female_data[column].std()
                female_synthetic = np.abs(np.random.normal(loc=female_mean, scale=female_std, size=int(0.8 * n_samples)))
                synthetic_data[column] = np.concatenate((male_synthetic, female_synthetic))
            else:  # Categorical features
                # For males
                male_counts = male_data[column].value_counts(normalize=True)
                male_synthetic = np.random.choice(male_counts.index, size=int(0.2 * n_samples), p=male_counts.values)
                # For females
                female_counts = female_data[column].value_counts(normalize=True)
                female_synthetic = np.random.choice(female_counts.index, size=int(0.8 * n_samples), p=female_counts.values)
                synthetic_data[column] = np.concatenate((male_synthetic, female_synthetic))

    # Combine the data into a new DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df['Gender'] = gender_imbalanced
    
    # Display the new distribution of 'Gender'
    #print("New Gender Distribution:")
    #print(synthetic_df['Gender'].value_counts())

    # Save the synthetic dataset to a new CSV file
    synthetic_df.to_csv('synthetic_dataset.csv', index=False)


create_synthetic_dataset(1000)