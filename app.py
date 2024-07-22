
from flask import Flask, render_template, redirect, url_for, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import replicate
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

# Load the saved model and column names
model = joblib.load('diabetes_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Set Replicate API Token
REPLICATE_API_TOKEN = "r8_X5Vgg7bJZSrb5WqBj7DfPXPcA7G8T9i0ED1R4"
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/info')
def info():
    return render_template("info.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/symptoms")
def symptoms():
    return redirect("https://www.who.int/health-topics/diabetes?gad_source=1&gclid=CjwKCAjw4_K0BhBsEiwAfVVZ_1sLwEwr3CJYIqAq0ZxaCbOZuEQVRWm_lQQql87wM-DABt9322YXDRoCxu4QAvD_BwE#tab=tab_2")

@app.route("/tips")
def tips():
    return redirect("https://www.diabetes.org.uk/guide-to-diabetes/enjoy-food/eating-with-diabetes/10-ways-to-eat-well-with-diabetes")

@app.route("/diet")
def diet():
    return redirect("https://www.diabetes.org.uk/guide-to-diabetes/enjoy-food/eating-with-diabetes/what-is-a-healthy-balanced-diet")

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
            return render_template('diab_result.html', 
                                   probability=round(probability[0], 2),
                                   age=input_data['age'],
                                   bmi=input_data['bmi'],
                                   blood_glucose_level=input_data['blood_glucose_level'],
                                   HbA1c_level=input_data['HbA1c_level'],
                                   hypertension=input_data['hypertension'],
                                   heart_disease=input_data['heart_disease'],
                                   gender=input_data['gender'],
                                   smoking_history=input_data['smoking_history'])
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred. Please check the logs."


# Helper function to generate LLaMA2 response
def generate_llama2_response(messages):
    string_dialogue = "You are a helpful assistant. Your purpose is to provide guidance based on the user's health data and model predictions. "
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    response_generator = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={"prompt": f"{string_dialogue} Assistant: ", "temperature": 0.1, "top_p": 0.9,
               "max_length": 10000, "repetition_penalty": 1}
    )

    response = ""
    for resp in response_generator:
        response += resp

    response += "\n\nPlease note: Always seek professional medical care for accurate diagnosis and treatment."

    return response

@app.route('/get_advice', methods=['POST'])
def get_advice():
    if request.method == 'POST':
        try:
            data = request.json
            user_data = data.get('user_data', {})
            model_prediction = data.get('model_prediction', "")

            # Prepare messages for LLaMA2 model
            messages = [{"role": "user", "content": f"Based on the user's data: {user_data} and the model's prediction: {model_prediction}, provide advice."}]

            # Get the response from the LLaMA2 model
            response = generate_llama2_response(messages)

            return jsonify({"response": response})
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": "An error occurred. Please check the logs."})


@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        try:
            data = request.json
            messages = data.get('messages', [])
            user_data = data.get('user_data', {})
            model_prediction = data.get('model_prediction', "")

            # Add user data and model prediction to messages if provided
            if user_data and model_prediction:
                messages.append({"role": "user", "content": f"Based on the user's data: {user_data} and the model's prediction: {model_prediction}, provide advice."})
            
            response = generate_llama2_response(messages)
            return jsonify({"response": response})
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": "An error occurred. Please check the logs."})

@app.route('/show_advice')
def show_advice():
    advice = request.args.get('advice', '')
    return render_template('advice.html', advice=advice)

if __name__ == "__main__":
    app.run(debug=True)





