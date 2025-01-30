# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Data
st.title("Titanic Dataset Logistic Regression Analysis")

train_data = pd.read_csv("Titanic_train.csv")
test_data = pd.read_csv("Titanic_test.csv")

st.subheader("Dataset Shapes")
st.write("Test Data Shape:", test_data.shape)
st.write("Train Data Shape:", train_data.shape)

st.subheader("Preview of Data")
st.write("Test Data Head:")
st.write(test_data.head())
st.write("Train Data Head:")
st.write(train_data.head())

# Checking for missing values
st.subheader("Missing Values")
st.write(train_data.isnull().sum())
st.write(test_data.isnull().sum())

# Visualizing missing values
st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots()
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)

# Handling missing values
train_data.dropna(inplace=True)

# Converting categorical features
st.subheader("Converting Categorical Features")
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
train_data["Embarked"] = train_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

st.write("Processed Train Data:")
st.write(train_data.head())

# Splitting data
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert categorical features using one-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure train and test have the same features
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

st.subheader("Training and Testing Split")
st.write("Training Data Shape:", X_train.shape)
st.write("Testing Data Shape:", X_test.shape)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Accuracy")
st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# User Input Features for Prediction
st.sidebar.header('User Input Parameters')

def user_input_features():
    Pclass = st.sidebar.selectbox('Pclass', (1, 2, 3))
    Sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    Age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    SibSp = st.sidebar.selectbox('SibSp', (0, 1, 2, 3, 4, 5))
    Parch = st.sidebar.selectbox('Parch', (0, 1, 2, 3, 4, 5))
    Fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    Embarked = st.sidebar.selectbox('Embarked', ('S', 'C', 'Q'))
    
    data = {'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# Preprocess user input
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Ensure the user input has the same features as the training data
df = pd.get_dummies(df, drop_first=True)
df = df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Survived' if prediction[0] == 1 else 'Not Survived')

st.subheader('Prediction Probability')
st.write(prediction_proba)