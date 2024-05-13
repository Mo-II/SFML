import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from Dataset.pre_processing import pre_processing


def plot_loss(history):
    plt.plot(history.loss_curve_, label='Training Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file = "./Dataset/DataSet.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = pre_processing(file)

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
