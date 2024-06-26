import pandas as pd
from sklearn.model_selection import train_test_split
import os


def make_categorical(data):
    # Convert categorical variables to numerical variables
    data['Gender'] = pd.Categorical(data['Gender']).codes
    data['family_history_with_overweight'] = pd.Categorical(data['family_history_with_overweight']).codes
    data['FAVC'] = pd.Categorical(data['FAVC']).codes
    data['CAEC'] = pd.Categorical(data['CAEC']).codes
    data['SMOKE'] = pd.Categorical(data['SMOKE']).codes
    data['SCC'] = pd.Categorical(data['SCC']).codes
    data['CALC'] = pd.Categorical(data['CALC']).codes
    data['MTRANS'] = pd.Categorical(data['MTRANS']).codes
    data['NObeyesdad'] = pd.Categorical(data['NObeyesdad']).codes

    return data

def pre_processing_classification(path):
    print(path)
    data = pd.read_csv(path)

    #print(data.shape)
    data = make_categorical(data)
    

    # Split the dataset into input features and the target variable
    X = data.drop('NObeyesdad', axis=1)
    X = X.drop('Weight', axis=1)
    Y = data['NObeyesdad']

    
    # Split the dataset into training and testing sets
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Example
#root = os.getcwd()
#file = root + "\\Dataset.csv"
#X_train, X_val, X_test, y_train, y_val, y_test = pre_processing(file)

def pre_processing_regression(path):

    data = pd.read_csv(path)

    data = make_categorical(data)

    X = data.drop('NObeyesdad', axis=1)
    X = X.drop('Weight', axis=1)

    Y = data['Weight']

    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test   