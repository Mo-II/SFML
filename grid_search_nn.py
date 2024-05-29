from Dataset.pre_processing import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import f1_score, make_scorer



def grid_search_nn(max_iter):

    file = "./Dataset/Dataset.csv"

    X_train, X_test, X_val, y_train, y_val, y_test = pre_processing(file)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,),(150,),(100,50),(150,75),(50,25)],  # Tuple specifying the number of neurons in each hidden layer
        'activation': ['relu', 'tanh'],  # Activation function options
        'solver': ['adam', 'sgd'],  # Optimization solver options
        'alpha': [0.0001, 0.001, 0.01],  # L2 penalty parameter
        'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule for weight updates
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],  # Initial learning rate
        'batch_size': [8, 16, 32, 64],  # Size of minibatches for stochastic optimizers
        'momentum': [0.9, 0.95, 0.99]  # Momentum for SGD optimizer
    }

    classifier = MLPClassifier(max_iter=max_iter, early_stopping=True, n_iter_no_change=10,random_state=42)

    # Create a scorer for multiclass classification
    scorer = make_scorer(f1_score, average='macro')

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(classifier, param_grid, cv=3, verbose=2, scoring=scorer)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the test set
    #test_score = grid_search.best_estimator_.score(X_test, y_test)
    #print("Best Model Test Score:", test_score)

    # Extract information from Grid Search results
    grid_search_results = grid_search.cv_results_

    # Convert cv_results_ to a DataFrame
    results_df = pd.DataFrame(grid_search_results)

    # Save the DataFrame to a CSV file
    results_df.to_csv('grid_search/grid_search_results_nn.csv', index=False)


grid_search_nn(100)
