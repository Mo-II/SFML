from sklearn.model_selection import RandomizedSearchCV
from Dataset.pre_processing import *
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import os
import pickle


file = "./Dataset/DataSet.csv"

X_train, X_test, X_val, y_train, y_val, y_test = pre_processing(file)

param_grid = {
    'hidden_layer_sizes': [(50,),(100,),(150,),(100,50),(150,75),(50,25)],  # Tuple specifying the number of neurons in each hidden layer
    'activation': ['relu', 'tanh'],  # Activation function options
    'solver': ['adam', 'sgd'],  # Optimization solver options
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty parameter
    'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule for weight updates
    'learning_rate_init': [0.0001 ,0.001, 0.01, 0.1],  # Initial learning rate
    'batch_size': [8 ,16, 32, 64],  # Size of minibatches for stochastic optimizers
    'momentum': [0.9, 0.95, 0.99],  # Momentum for SGD optimizer
}

classifier = MLPClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=10)

# Perform grid search with cross-validation
grid_search = GridSearchCV(classifier, param_grid, cv=3, verbose=2)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Save the grid search results to a file
with open('grid_search_results.pkl', 'wb') as f:
    pickle.dump(grid_search.cv_results_, f)

# Evaluate the best model on the test set
test_score = grid_search.best_estimator_.score(X_test, y_test)
print("Best Model Test Score:", test_score)

# Load the grid search results from the file
with open('grid_search_results.pkl', 'rb') as f:
    grid_search_results = pickle.load(f)

# Access and analyze the grid search results
print(grid_search_results)