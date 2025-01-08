import pickle

# Assuming 'model' is your trained model
with open('models/your_model.pkl', 'wb') as f:
    pickle.dump(model, f)
