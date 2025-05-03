import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📚 Student Exam Score Predictor")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("StudentsPerformance.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    encoder = LabelEncoder()
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        data[col] = encoder.fit_transform(data[col])

    X = data[['math score', 'reading score', 'writing score']]
    y_math = X['math score']
    y_reading = X['reading score']
    y_writing = X['writing score']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y_math, y_reading, y_writing, scaler

# Train models
def train_models(X, y_math, y_reading, y_writing):
    X_math = X[['reading score', 'writing score']]
    X_reading = X[['math score', 'writing score']]
    X_writing = X[['math score', 'reading score']]

    X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(X_math, y_math, train_size=0.7)
    X_reading_train, X_reading_test, y_reading_train, y_reading_test = train_test_split(X_reading, y_reading, train_size=0.7)
    X_writing_train, X_writing_test, y_writing_train, y_writing_test = train_test_split(X_writing, y_writing, train_size=0.7)

    math_model = LinearRegression().fit(X_math_train, y_math_train)
    reading_model = LinearRegression().fit(X_reading_train, y_reading_train)
    writing_model = LinearRegression().fit(X_writing_train, y_writing_train)

    math_R2 = math_model.score(X_math_test, y_math_test)
    reading_R2 = reading_model.score(X_reading_test, y_reading_test)
    writing_R2 = writing_model.score(X_writing_test, y_writing_test)

    return math_model, reading_model, writing_model, math_R2, reading_R2, writing_R2

# Load and prepare data
data = load_data()
X, y_math, y_reading, y_writing, scaler = preprocess_data(data)
math_model, reading_model, writing_model, math_R2, reading_R2, writing_R2 = train_models(X, y_math, y_reading, y_writing)

st.markdown("Enter any two subject scores to predict the third one using Linear Regression.")

subject = st.selectbox("Which subject do you want to predict?", ["Math", "Reading", "Writing"])

if subject == "Math":
    read = st.number_input("Reading Score:", min_value=0, max_value=100)
    write = st.number_input("Writing Score:", min_value=0, max_value=100)
    if st.button("Predict Math Score"):
        input_data = np.array([[0, read, write]])
        input_scaled = scaler.transform(input_data)
        pred = math_model.predict([[input_scaled[0][1], input_scaled[0][2]]])[0]
        st.success(f"Predicted Math Score: {pred:.2f}")
        st.info(f"Model R² Score: {math_R2:.3f}")

elif subject == "Reading":
    math = st.number_input("Math Score:", min_value=0, max_value=100)
    write = st.number_input("Writing Score:", min_value=0, max_value=100)
    if st.button("Predict Reading Score"):
        input_data = np.array([[math, 0, write]])
        input_scaled = scaler.transform(input_data)
        pred = reading_model.predict([[input_scaled[0][0], input_scaled[0][2]]])[0]
        st.success(f"Predicted Reading Score: {pred:.2f}")
        st.info(f"Model R² Score: {reading_R2:.3f}")

elif subject == "Writing":
    math = st.number_input("Math Score:", min_value=0, max_value=100)
    read = st.number_input("Reading Score:", min_value=0, max_value=100)
    if st.button("Predict Writing Score"):
        input_data = np.array([[math, read, 0]])
        input_scaled = scaler.transform(input_data)
        pred = writing_model.predict([[input_scaled[0][0], input_scaled[0][1]]])[0]
        st.success(f"Predicted Writing Score: {pred:.2f}")
        st.info(f"Model R² Score: {writing_R2:.3f}")
