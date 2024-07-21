import joblib
import pandas as pd

# Load the saved model and column names
model = joblib.load('diabetes_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Example input data
input_data = pd.DataFrame({
    'age': [59],
    'hypertension': [1],
    'heart_disease': [0],
    'bmi': [16],
    'HbA1c_level': [7.4],
    'blood_glucose_level': [300],
    'gender': ['Female'],
    'smoking_history': ['never']
})


# Function to preprocess input data
def preprocess_input(data):
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


# Preprocess the input data
input_data_preprocessed = preprocess_input(input_data, model_columns)

# Make prediction
prediction = model.predict(input_data_preprocessed)
prediction_proba = model.predict_proba(input_data_preprocessed)[:, 1]

input_data_preprocessed = preprocess_input(input_data, model_columns)
print(f"Preprocessed data: {input_data_preprocessed.head()}") 

# Output the result
print(f"Prediction: {prediction[0]}")
print(f"Probability of having diabetes: {prediction_proba[0] * 100:.2f}%")


