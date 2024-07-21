

from flask import Flask, render_template, redirect, url_for, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

# Load the saved model and column names
model = joblib.load('diabetes_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

GEMINI_API_KEY = 'your_gemini_api_key'

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/info')
def info():
    return render_template("info.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

# Function to preprocess input data
def preprocess_input(data, model_columns):
    # Recategorize 'smoking_history'
    def recategorize_smoking(smoking_status):
        if smoking_status in ['never', 'No Info']:
            return 'non-smoker'
        elif smoking_status == 'current':
            return 'current_smoker'
        else:
            return 'past_smoker'

    # Apply recategorization
    data['smoking_history'] = data['smoking_history'].apply(recategorize_smoking)

    # Create one-hot encoded columns for 'smoking_history'
    data = pd.get_dummies(data, columns=['smoking_history'], drop_first=False)

    # Convert gender to binary columns
    data['gender_Male'] = (data['gender'] == 'Male').astype(int)
    data['gender_Other'] = (data['gender'] == 'Other').astype(int)
    data = data.drop(columns=['gender'])  # Drop the original gender column

    # Ensure all required columns are present
    for column in model_columns:
        if column not in data.columns:
            data[column] = 0
    data = data[model_columns]

    return data


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Gather input data
            input_data = {
                'age': float(request.form['age']),
                'bmi': float(request.form['bmi']),
                'blood_glucose_level': float(request.form['blood_glucose_level']),
                'HbA1c_level': float(request.form['HbA1c_level']),
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'gender': request.form['gender'],
                'smoking_history': request.form['smoking_history']
            }

            # Convert input data to DataFrame
            input_data_df = pd.DataFrame([input_data])

            # Preprocess the input data
            input_data_preprocessed = preprocess_input(input_data_df, model_columns)

            # Make prediction
            prediction_proba = model.predict_proba(input_data_preprocessed)[:, 1]

            probability = prediction_proba * 100  
            return render_template('diab_result.html', probability=round(probability[0],2))
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred. Please check the logs."
        

@app.route('/get_assistance', methods=['POST'])
def get_assistance():
    data = request.json
    result = data['result']
    genai_prompt = f"Based on the diabetes detection result of {result}, provide personalized assistance."

    try:
        response = requests.post(
            'https://api.gemini.com/v1/chat',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {GEMINI_API_KEY}'
            },
            json={
                'model': 'text-davinci-002',  # or any other model you are using
                'prompt': genai_prompt,
                'max_tokens': 150
            }
        )
        response_data = response.json()
        answer = response_data['choices'][0]['text'].strip()

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error fetching GenAI response: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == "__main__":
    app.run(debug=True)

