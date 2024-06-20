import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
