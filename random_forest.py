from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from Dataset.pre_processing import pre_processing_regression
from Dataset.pre_processing import make_categorical

def evaluate_regressor(regressor, X, y):
    # Make predictions
    y_pred = regressor.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)  # Measures the mean-squared-error between the predicted values and the actual values

    # Calculate R-squared
    r2 = r2_score(y, y_pred)  # How well the model can explain the variance in weight => higher r2_score, better at predicting weight

    return mse, r2

def plot_learning_curves(estimator, X_train, y_train, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5, 
                                                            scoring='mean_squared_error', 
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation error")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file = "./Dataset/ObesityDataSet.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = pre_processing_regression(file)
    # In pre_processing_regression, the NObeyesdad and Weight columns are removed, and the weight column is used as the target value
    # The NObeyesdad is removed because it has a high correlation with Weight

    # n_estimators: the number of decision trees in the forest
    # max_depth: the maximum depth of each tree in the forest
    # min_samples_split: the minimum number of samples required to split an internal node => can help prevent overfitting
    # min_samples_leaf: minimum number of samples required at a leaf node, can also help prevent overfitting
    # max_features: the maximum number of features considered when splitting a node
    # bootstrap: whether bootstrap samples are used when building trees; if False, the whole dataset is used to build each tree
    # random_state: sets a random seed for reproducibility
    # criterion: specifies the function to measure the quality of a split: mean squared error or mean absolute error
    # warm_start: allows previous solutions/fits to be reused

    # These parameters are the result of the grid search we conducted on the supercomputer.
    bootstrap = False
    max_depth = 30
    max_features = 'sqrt'
    min_samples_leaf = 1
    min_samples_split = 8
    n_estimators = 90
    random_state = 42

# Creating the model
    regressor = RandomForestRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, 
                                    max_features=max_features, 
                                    bootstrap=bootstrap,
                                    random_state=random_state)

    # Fit the model
    regressor.fit(X_train, y_train)

    # Plot learning curves
    val_mse, val_r2 = evaluate_regressor(regressor, X_val, y_val)

    print("Validation Mean Squared Error:", val_mse)
    print("Validation R-squared:", val_r2)

    feature_importances = regressor.feature_importances_

    # Pair feature importances with feature names
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print feature importances
    print("Feature Importances:")
    print(feature_importance_df)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()

    # The graph that is shown visualizes the feature importance of the attributes on the target value 'Weight'. It is intuitive that 'Height' has a high feature importance because height is directly correlated to the weight of a person, which is an obvious relationship that the model confirms.
    # This is why we decided to highlight the 2nd most important feature, which is 'Family history with overweight', which indicates that genetics and environmental factors play a significant role in whether a person is obese/has CVD risk or not, which we did not expect to be the case.