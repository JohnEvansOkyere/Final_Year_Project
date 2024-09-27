import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('processed_numeric.csv')  # Update with your dataset path

# Encode categorical variables if needed
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save the label encoders for later use
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

# Define features and target
X = df.drop(columns='Cluster')  # Replace 'Cluster' with your target variable
y = df['Cluster']  # Replace 'Cluster' with your target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model using pickle
with open('Frandom_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

print("Model training complete and saved as 'Frandom_forest_model.pkl'.")
