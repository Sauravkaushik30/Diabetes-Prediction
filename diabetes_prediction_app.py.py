#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up the title and introduction
st.title('Diabetes Prediction System')
st.write('This app predicts whether a person is diabetic or not based on medical input data.')

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Data preprocessing
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Define a function to make predictions
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape for a single prediction
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Standardize the input data
    standardized_input_data = scaler.transform(input_data_as_numpy_array)
    
    # Make the prediction
    prediction = classifier.predict(standardized_input_data)
    
    if prediction[0] == 0:
        return 'The person is NOT diabetic.'
    else:
        return 'The person is diabetic.'

# User input for the prediction system
st.sidebar.header('Enter Medical Data')

# Create input fields for each feature
Pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
Glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=100, step=1)
BloodPressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70, step=1)
SkinThickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20, step=1)
Insulin = st.sidebar.number_input('Insulin Level', min_value=0, max_value=900, value=79, step=1)
BMI = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0, step=0.1)
DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
Age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30, step=1)

# Create a button to submit the input data
if st.button('Predict'):
    # Store the input data in a tuple
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    
    # Call the prediction function
    diagnosis = diabetes_prediction(input_data)
    
    # Display the result
    st.success(diagnosis)

# Display accuracy of the model on test data
test_accuracy = accuracy_score(y_test, classifier.predict(x_test))
st.write(f"Model Accuracy on Test Data: {test_accuracy * 100:.2f}%")


# In[ ]:




