from Dataset.pre_processing import pre_processing
import lightgbm as lgb
from sklearn.metrics import classification_report
import os

root = os.getcwd()
file = root + "\\Dataset\\Dataset.csv"

X_train, X_Val, X_test, y_train, y_val, y_test = pre_processing(file)

print (X_train.shape)

params = {
'objective': 'multiclass',
'boosting_type': 'gbdt',
'num_leaves': 31,
'learning_rate': 0.05,
"num_class": 7
#'feature_fraction': 0.9,
}
classifier = lgb.LGBMClassifier(**params)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))