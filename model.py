import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load the data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Remove duplicates
data = data.drop_duplicates()

# Convert gender column to one-hot encoding
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Converge smoking history into categories
def converge_smoking_status(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

data['smoking_history'] = data['smoking_history'].apply(converge_smoking_status)

# Perform one-hot encoding on the smoking history column
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Define features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Apply SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize and train the classifier
rf_classifier = RandomForestClassifier(criterion='entropy', max_depth=4, max_features='log2', n_estimators=100, random_state=2)
rf_classifier.fit(X_train_res, y_train_res)

# Save the trained model and column names
joblib.dump(rf_classifier, 'diabetes_rf_model.pkl')
joblib.dump(X_train_res.columns, 'model_columns.pkl')

