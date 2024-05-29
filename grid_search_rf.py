from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from Dataset.pre_processing import *
import pandas as pd

def grid_search_rf():

    file = "./Dataset/DataSet.csv"

    X_train, X_test, X_val, y_train, y_val, y_test = pre_processing(file)


    # Define the parameter grid
    param_grid = {
    'n_estimators': [50, 100],#, 200],
    'max_depth': [None, 10]#, 20, 30]
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'bootstrap': [True, False],
    #'max_features': ['auto', 'sqrt', 'log2']
    }

    regressor = RandomForestRegressor()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3, verbose=2, scoring='mean_squared_error')

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
    results_df.to_csv('grid_search/grid_search_results_rf.csv', index=False)

grid_search_rf()