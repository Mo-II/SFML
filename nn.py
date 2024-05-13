import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Further split X_temp and y_temp into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_loss(history):
    plt.plot(history.loss_curve_, label='Training Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file = "./Dataset/DataSet.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = datasplitter(file)

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
