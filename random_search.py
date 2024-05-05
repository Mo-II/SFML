from sklearn.model_selection import RandomizedSearchCV
from Dataset.pre_processing import *
import lightgbm as lgb
from sklearn.metrics import classification_report
import os

root = os.getcwd()
file = root + "\\Dataset\\ObesityDataSet.csv"

X_train, X_test, y_train, y_test = pre_processing(file)

param_grid = {
    'num_leaves': [20, 30, 40, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0]
}

classifier = lgb.LGBMClassifier(objective='multiclass')

random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', verbose=2, random_state=42)
random_search.fit(X_train, y_train)

print("Best Hyperparameters:", random_search.best_params_)

best_classifier = lgb.LGBMClassifier(**random_search.best_params_)
best_classifier.fit(X_train, y_train)

predictions = best_classifier.predict(X_test)
print(classification_report(y_test, predictions))
