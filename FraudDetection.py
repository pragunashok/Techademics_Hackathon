import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the training dataset
train_data = pd.read_csv('AllApplications.csv')

# List of columns to ignore
ignore_cols = ['Entry', "Mother's Name", "Mother's Occupation", "Mother's Education",
               "Father's Name", "Father's Occupation", "Father's Education", 'Writing']

# Drop the ignored columns
train_data = train_data.drop(ignore_cols, axis=1)

# Separate numerical and categorical columns
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Separate the features and target variable
X = train_data.drop('Fraudulent', axis=1)
y = train_data['Fraudulent']

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Load the test dataset
test_data = pd.read_csv('AllApplications.csv')

# Drop the ignored columns from the test set
test_data = test_data.drop(ignore_cols, axis=1)

# Separate numerical and categorical columns
numerical_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = test_data.select_dtypes(include=['object']).columns

# Preprocess the test data
X_test = test_data.drop('Fraudulent', axis=1)
X_test = preprocessor.transform(X_test)

# Predict the probabilities for the test set
y_prob = clf.predict_proba(X_test)[:, 1]

# Add the predicted probabilities to the test dataset
test_data['Fraudulent_probability'] = y_prob

# Print the test dataset with predictions
#print(test_data)
print(test_data[['Legal Name','Fraudulent_probability','Fraudulent']])