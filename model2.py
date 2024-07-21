

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import joblib
# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Separate features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Define the sampling strategy
sampling_strategy = {0: 10000, 1: 8500}

# Initialize the RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

# Apply the undersampler to the data
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combine the resampled features and target into a single DataFrame
balanced_data = pd.concat([X_resampled, y_resampled], axis=1)

# Shuffle the balanced dataset to mix the instances
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Print the new class distribution to verify
print("New class distribution in the balanced dataset:")
print(balanced_data['diabetes'].value_counts())

                     
warnings.filterwarnings("ignore")
# Recategorize 'smoking_history' column
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'never'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'former'

# Apply the function to the 'smoking_history' column
balanced_data['smoking_history'] = balanced_data['smoking_history'].apply(recategorize_smoking)

# Ordinal Encoding for 'smoking_history'
smoking_mapping = {'never': 0, 'former': 1, 'current': 2}
balanced_data['smoking_history'] = balanced_data['smoking_history'].map(smoking_mapping)

# One-Hot Encoding for 'gender'
one_hot = pd.get_dummies(balanced_data, columns=['gender'], drop_first=True)

# Check the first few rows of the processed data
print(one_hot.head())

# Check the column names to ensure encoding is done correctly
print(one_hot.columns)
# _hot = pd.get_dummies(balanced_data, columns=['smoking_history', 'gender'], drop_first=True)

# Check the first few rows of the processed data
print(one_hot.head())

# Check the column names to ensure encoding is done correctly
print(one_hot.columns)

X_processed = one_hot.drop('diabetes', axis=1)
y_processed = one_hot['diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
best_rf_clf = RandomForestClassifier(criterion='gini', max_depth=None, max_features='sqrt',
                                     min_samples_leaf=4, min_samples_split=2, n_estimators=300,
                                     random_state=2)
best_rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = best_rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{report}')

# Save the model to a .pkl file
joblib.dump(best_rf_clf, 'best_rf_model.pkl')