import pickle
import torch
from sklearn.metrics import accuracy_score

def load_random_forest_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_random_forest(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def load_simple_nn_model(path, input_size, hidden_size, output_size):
    model = SimpleNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(path))
    return model

def evaluate_simple_nn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test).float())
    _, predictions = torch.max(outputs, 1)
    return accuracy_score(y_test, predictions.numpy())
