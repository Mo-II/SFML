from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from Dataset.pre_processing import pre_processing
from Dataset.pre_processing import make_categorical


def evaluate_regressor(regressor, X, y):
    # Make predictions
    y_pred = regressor.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred) #meet de mean-squared-error tussen de voorspelde waarden en de werkelijke waarden

    # Calculate R-squared
    r2 = r2_score(y, y_pred) #Hoe goed het model de variatie in gewicht kan verklaren => hogere r2_score, beter in het voorspellen van het gewicht

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
    file = "./Dataset/Dataset.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = pre_processing(file)

    #n_estimators: is het aantal decision trees in de forest
    #max_depth: staat voor de maximum diepte van elke boom in de forest
    #min_samples_split: is het minimum aantal samples nodig om een interne node te splitsen => kan helpen overfitten te voorkomen
    #min_samples_leaf: minimum aantal samples nodig bij een leaf node, kan ook helpen om overfitting te voorkomen
    #max_features: Staat voor het maximum aantal features die in acht worden genomen wanneer een node wordt gesplitst
    #bootstrap: Controleert of bootstrap samples worden gebruikt bij het bouwen van de trees, wanneer False, wordt elke tree getrained op de hele dataset
    #random_state: Set een random seed voor reproducibility
    #cirterion: specifieert de functie voor het meten van de kwaliteit van een split: mean squared error of mean absolute error
    #warm_start: Staat toe om vorige solutions/fits te hergebruiken
    regressor = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10,
                                       min_samples_leaf=1, max_features=15, bootstrap=True,
                                       random_state=42)

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