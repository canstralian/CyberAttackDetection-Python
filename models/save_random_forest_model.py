import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to the /models directory
model_path = "models/random_forest_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Random Forest model saved to {model_path}")
