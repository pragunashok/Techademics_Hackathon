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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Load the test dataset
test_data = pd.read_csv('AllApplicationsTest.csv')


new_row_list = ['6','Olivia Chen','25-09-2001','(678) 901-2345','678 Willow Court','Female','Jennifer Chen','Dentist','Doctor of Dental Surgery','(543) 210-9876','Steven Chen','Professor','Ph.D.','(789) 456-1230','2017-2021','Willowbrook High School','3.8','Science Club','[Personal essay text]','Not paid','1430','27','15','5','97','6','FALSE']





new_row = {'Entry':new_row_list[0],'Legal Name':new_row_list[1],'DOB':new_row_list[2],'Contact Info':new_row_list[3],'Residence':new_row_list[4],'Gender':new_row_list[5],"Mother's Name":new_row_list[6],"Mother's Occupation":new_row_list[7],"Mother's Education":new_row_list[8],"Mother's Phone":new_row_list[9],"Father's Name":new_row_list[10],"Father's Occupation":new_row_list[11],"Father's Education":new_row_list[12],"Father's Phone":new_row_list[13],"Education Dates":new_row_list[14],'Secondary School':new_row_list[15],'GPA':new_row_list[16],'Activities':new_row_list[17],'Writing':new_row_list[18],'Fees Status':new_row_list[19],'SAT Score (1600)':new_row_list[20],'ACT Score (36)':new_row_list[21],'AP Score (15)':new_row_list[22],'IB Score (7)':new_row_list[23],'TOEFL Score (120)':new_row_list[24],'IELTS Score (9)':new_row_list[25],'Fraudulent':'FALSE'}                                                  
#new_row = {'Entry':'6','Legal Name':'Olivia Chen','DOB':'25-09-2001','Contact Info':'(678) 901-2345','Residence':'678 Willow Court','Gender':'Female',"Mother's Name":'Jennifer Chen',"Mother's Occupation":'Dentist',"Mother's Education":'Doctor of Dental Surgery',"Mother's Phone":'(543) 210-9876',"Father's Name":'Steven Chen',"Father's Occupation":'Professor',"Father's Education":'Ph.D.',"Father's Phone":'(789) 456-1230',"Education Dates":'2017-2021','Secondary School':'Willowbrook High School','GPA':'3.8','Activities':'Science Club','Writing':'[Personal essay text]','Fees Status':'Not paid','SAT Score (1600)':'1430','ACT Score (36)':'27','AP Score (15)':'15','IB Score (7)':'5','TOEFL Score (120)':'97','IELTS Score (9)':'6','Fraudulent':'FALSE'}                                                  
test_data = test_data.append(new_row, ignore_index=True)
#print(test_data.iloc[-1])
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

final_output = test_data.iloc[-1]['Fraudulent_probability']




print(test_data.iloc[-1][['Legal Name','Fraudulent_probability']])
