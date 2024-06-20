import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset (assuming the dataset is named 'data.csv')
heart_data = pd.read_csv('content\heart_disease_data.csv')

# Display the first few rows of the dataset
print(heart_data.head())

# Separating the features and target
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Training the Logistic Regression model with the training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test data: {accuracy:.2f}')

# Save the trained model to a file
model_filename = 'heart_disease_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model saved as {model_filename}')
