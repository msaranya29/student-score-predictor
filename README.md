# 🎓 Student Exam Score Predictor

A simple web app built with **Streamlit** and **Scikit-learn** that predicts a student's score in one subject (Math, Reading, or Writing) based on the other two scores using **Linear Regression**.

---

## 📌 Features

- Predict **Math**, **Reading**, or **Writing** score.
- Interactive UI built with **Streamlit**.
- Models trained using **Linear Regression**.
- Display of model accuracy using **R² score**.

---

## 🧠 How It Works

The app uses a cleaned version of the [StudentsPerformance.csv](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset. It encodes categorical features and scales numerical data before training separate linear regression models:

- Math Score = f(Reading, Writing)
- Reading Score = f(Math, Writing)
- Writing Score = f(Math, Reading)

---

## 🚀 Run the App Locally

### 1. Clone the repository


  ```bash git clone https://github.com/msaranya29/student-score-predictor.git cd student-score-predictor ``` 

### 2. Install Dependencies 
pip install -r requirements.txt


### 3. Start the Streamlit App
streamlit run app.py


📊 Dataset Source : Kaggle Dataset



