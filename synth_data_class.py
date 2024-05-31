import random
import pandas as pd

trainingsset = pd.read_csv("Dataset/ObesityDataSet.csv")

unique_values = trainingsset['NObeyesdad'].unique()
print(unique_values)

labels = trainingsset['NObeyesdad'].unique()

mislabel_percentages = [0.05, 0.1, 0.15, 0.2]

for mislabel_percentage in mislabel_percentages:
    mislabeled_df = trainingsset.copy()

    for i in range(len(mislabeled_df)):
        random_num = random.random()

        if random_num < mislabel_percentage:
            current_label = mislabeled_df.iloc[i]['NObeyesdad']
            new_label = random.choice([l for l in labels if l != current_label])

            mislabeled_df.iloc[i, mislabeled_df.columns.get_loc('NObeyesdad')] = new_label

    filename = f"Dataset/Dataset_mislabeled__class_{int(mislabel_percentage*100)}%.csv"
    mislabeled_df.to_csv(filename, index=False)
    print(f"Mislabeled dataset met {mislabel_percentage*100}% mislabeling weggeschreven naar {filename}")