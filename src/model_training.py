import pickle
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_simple_nn(X_train, y_train, input_size, hidden_size, output_size):
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Dummy training loop
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train).float())
        loss = criterion(outputs, torch.tensor(y_train).long())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'models/simple_nn_model.pth')
    return model
