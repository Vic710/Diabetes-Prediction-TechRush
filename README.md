Diabetes Prediction and Personalized Advice Application
Overview
This project is a web application that predicts the likelihood of diabetes based on user-provided health data and offers personalized advice using the Gemini API. The application uses a trained Random Forest model for prediction and the Gemini API to generate context-aware health advice.

Features
Diabetes Prediction: Users can input their health data to get a prediction of their likelihood of having diabetes.
Personalized Health Advice: The application provides personalized advice based on the prediction and user data using the Gemini API.
Informative Links: Users can access additional information about diabetes symptoms, tips, and diet plans through embedded links.
Interactive UI: The web interface is built using Flask and Bootstrap for a user-friendly experience.
Technologies Used
Backend:

Flask: A micro web framework for Python to build the web application.
Sklearn: For building and using the Random Forest model.
Joblib: For saving and loading the trained model and preprocessing steps.
Google Generative AI (Gemini API): For generating personalized health advice.
Frontend:

HTML, CSS, Bootstrap: For designing the web pages and ensuring responsiveness.
Deployment:

The application is deployed locally, and can be hosted on a cloud platform for broader accessibility.
Project Workflow
1. Data Collection
Users input their health data through a web form. The required fields include:

Age
BMI
Blood Glucose Level
HbA1c Level
Hypertension status
Heart Disease status
Gender
Smoking History


2. Data Preprocessing
The input data is preprocessed to match the format required by the Random Forest model:

Recategorization: The smoking history is recategorized into 'non-smoker', 'current smoker', and 'past smoker'.
One-Hot Encoding: Categorical variables such as gender and smoking history are converted to one-hot encoded columns.
Column Matching: Ensures all columns required by the model are present in the input data.

3. Prediction
The preprocessed data is passed to the Random Forest model to get the probability of having diabetes. The model was trained on a diabetes dataset and saved using Joblib.

4. Personalized Advice Generation
The user data and model prediction are sent to the Gemini API, which generates personalized health advice. The advice includes practical tips and recommendations based on the input data and prediction results.

5. Display Results
The prediction and advice are displayed on a results page, with HTML formatting preserved to enhance readability.
