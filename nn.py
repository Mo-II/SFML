
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()                        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def datasplitter(file_path):
    data = pd.read_csv(file_path)

    #Encode non numerical columns
    label_encoder = LabelEncoder()
    encoded_columns = data.select_dtypes(include=['object']).columns
    for column in encoded_columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    features = data.drop(columns=['NObeyesdad']).columns.tolist()

    X = data[features]
    y = data['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    file = "./Dataset/ObesityDataSet.csv"
    X_train, X_test, y_train, y_test = datasplitter(file)
    
    input_size = X_train.shape[1]
    hidden_size = 64
    num_classes = len(y_train.unique())
    model = NeuralNet(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Forward pass
        features = torch.tensor(X_train.values, dtype=torch.float)
        outputs = model(features)
        label = torch.tensor(y_train.values, dtype=torch.long)
        loss = criterion(outputs, label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training finished.')

    model.eval()
    # Testing the model
    with torch.no_grad():
        features = torch.tensor(X_test.values, dtype=torch.float)
        # Forward pass on test data
        outputs = model(features)
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)        
        # # Calculate accuracy
        correct = (predicted == y_test).sum().item()
        print(correct)