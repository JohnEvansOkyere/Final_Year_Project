import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Load the trained Random Forest model
with open('Frandom_forest_model.pkl', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have index.html created as explained earlier

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    laptop_brand = int(request.form['laptop_brand'])
    ram_gb = float(request.form['ram_gb'])
    storage = int(request.form['storage'])
    processor = int(request.form['processor'])
    condition = int(request.form['condition'])
    price = float(request.form['price'])
    region = int(request.form['region'])

    # Prepare the features for prediction
    features = np.array([[laptop_brand, ram_gb, storage, processor, condition, price, region]])

    # Predict the cluster
    prediction = rf_classifier.predict(features)[0]

    # Outputting the prediction result
    return f'This customer is grouped under cluster {prediction}. Based on this result, Jiji can recommend suitable products tailored to this customer’s preferences.'

