import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

hidden_layer_sizes = (64, 128)
activation = 'tanh'
batch_size = 32
learning_rate = 'adaptive'
learning_rate_init = 0.01
solver = 'adam'

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
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def plot_loss(history):
    plt.plot(history.loss_curve_, label='Training Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Modify plot_feature_importances function
def plot_permutation_importances(perm_importances, features):
    sorted_idx = perm_importances.importances_mean.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Permutation Importances")
    plt.bar(range(X_train.shape[1]), perm_importances.importances_mean[sorted_idx], align="center")
    plt.xticks(range(X_train.shape[1]), [features[i] for i in sorted_idx], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

if __name__ == "__main__":
    file = "./Dataset/ObesityDataSet.csv"
    X_train, X_test, y_train, y_test = datasplitter(file)

   # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the model
    input_size = X_train.shape[1]
    hidden_layer_sizes = (64, 128)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                      max_iter=500, 
                      learning_rate=learning_rate, 
                      learning_rate_init=learning_rate_init, 
                      activation=activation, 
                      solver=solver, 
                      batch_size=batch_size, 
                      random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')