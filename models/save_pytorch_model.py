import os
import torch
import torch.nn as nn
import torch.optim as optim

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define a simple neural network
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

# Initialize the model, loss function, and optimizer
input_size = 20
hidden_size = 50
output_size = 2
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Save the model to the /models directory
model_path = "models/simple_nn_model.pth"
torch.save(model.state_dict(), model_path)

print(f"PyTorch model saved to {model_path}")
