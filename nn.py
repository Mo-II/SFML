import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

def datasplitter(file_path):
    data = pd.read_csv(file_path)

    # Encode non numerical columns
    label_encoder = LabelEncoder()
    encoded_columns = data.select_dtypes(include=['object']).columns
    for column in encoded_columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    features = data.drop(columns=['NObeyesdad']).columns.tolist()

    X = data[features]
    y = data['NObeyesdad']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def plot_loss(history):
    plt.plot(history.loss_curve_, label='Training Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Modify plot_feature_importances function
def plot_permutation_importances(perm_importances, features):
    sorted_idx = perm_importances.importances_mean.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Permutation Importances")
    plt.bar(range(X_train.shape[1]), perm_importances.importances_mean[sorted_idx], align="center")
    plt.xticks(range(X_train.shape[1]), [features[i] for i in sorted_idx], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

if __name__ == "__main__":
    file = "./Dataset/DataSet.csv"
    X_train, X_test, y_train, y_test = datasplitter(file)

    # Initialize and train the neural network with early stopping
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)
    history = clf.fit(X_train, y_train)

    # Evaluate the model on validation set
    val_loss = clf.loss_
    print(f"Validation Loss: {val_loss}")

    # Evaluate the model on test set
    test_accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")

    # Plot the training loss
    plot_loss(history)

    # Calculate permutation importances
    perm_importances = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

    # Plot permutation importances
    plot_permutation_importances(perm_importances, X_train.columns)