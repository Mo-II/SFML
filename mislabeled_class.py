import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from Dataset.pre_processing import *
import seaborn as sns

datasets = {
    'original': "Dataset/ObesityDataSet.csv",
    'mislabeled_5': "Dataset/Mislabeled_class/Dataset_mislabeled__class_5%.csv",
    'mislabeled_10': "Dataset/Mislabeled_class/Dataset_mislabeled__class_10%.csv",
    'mislabeled_15': "Dataset/Mislabeled_class/Dataset_mislabeled__class_15%.csv",
    'mislabeled_20': "Dataset/Mislabeled_class/Dataset_mislabeled__class_20%.csv"
}

hidden_layer_sizes = (64, 128)
activation = 'tanh'
batch_size = 32
learning_rate = 'adaptive'
learning_rate_init = 0.01
solver = 'adam'

def train_and_evaluate_model(dataset, model_type='nn'):
    X_train, X_val, X_test, y_train, y_val, y_test = pre_processing_classification(dataset)
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'nn':
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                      max_iter=500, 
                      learning_rate=learning_rate, 
                      learning_rate_init=learning_rate_init, 
                      activation=activation, 
                      solver=solver, 
                      batch_size=batch_size, 
                      random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'rf' or 'nn'.")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    
    return metrics

results_rf = {}
results_nn = {}

for name, dataset in datasets.items():
    results_nn[name] = train_and_evaluate_model(dataset, model_type='nn')


def plot_results(results, title):
    df = pd.DataFrame(results).T
    df = df.reset_index().melt(id_vars='index', var_name='metric', value_name='score')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='index', y='score', hue='metric', data=df)
    plt.title(title)
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.ylim(0, 1)
    plt.show()

# Plot results
plot_results(results_rf, 'Performance of Random Forest under Different Mislabeling Conditions')
plot_results(results_nn, 'Performance of Neural Network under Different Mislabeling Conditions')