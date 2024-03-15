# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Pima Indians Diabetes Database
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
diabetes_data = pd.read_csv(url, names=names)

# Display the first few rows of the dataset
print("Diabetes Data:")
print(diabetes_data.head())

# Split the dataset into features (X) and target variable (y)
X = diabetes_data.drop('class', axis=1)
y = diabetes_data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict probability for a new sample
new_sample = [[3, 110, 120/80, 26, 0, 18, 0.627, 16]]  # Example health parameters
new_sample_scaled = scaler.transform(new_sample)
probability = rf_classifier.predict_proba(new_sample_scaled)[0][1]  # Probability of having diabetes
print("\nProbability of having diabetes:", probability)
