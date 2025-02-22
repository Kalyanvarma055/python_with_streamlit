import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("accident.csv")
    return df

df = load_data()

st.title("Accident Severity Prediction")
st.write("This app uses Logistic Regression to predict accident severity based on provided features.")

# Show dataset
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Data Preprocessing
# st.subheader("Data Preprocessing")
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Helmet_Used'] = df['Helmet_Used'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Seatbelt_Used'] = df['Seatbelt_Used'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Speed_of_Impact'].fillna(df['Speed_of_Impact'].median(), inplace=True)

# Visualizations
# st.subheader("Data Visualization")
# if st.checkbox("Show Correlation Heatmap"):
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    #st.pyplot(plt)

# Model Training
# st.subheader("Train Logistic Regression Model")

X = df[['Gender', 'Helmet_Used', 'Seatbelt_Used', 'Speed_of_Impact']]
y = df['Survived']  # Assuming 'Severity' is the target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# st.write(f"Model Accuracy: {accuracy:.2f}")
# st.text("Classification Report:")
# st.text(classification_report(y_test, y_pred))

# Prediction UI
st.subheader("Make a Prediction")
gender = st.radio("Select Gender", ["Male", "Female"])
helmet_used = st.radio("Helmet Used?", ["Yes", "No"])
seatbelt_used = st.radio("Seatbelt Used?", ["Yes", "No"])
speed_of_impact = st.number_input("Speed of Impact", min_value=0.0, step=0.1)

if st.button("Predict Survival"):
    input_data = np.array([[1 if gender == "Male" else 0, 1 if helmet_used == "Yes" else 0, 1 if seatbelt_used == "Yes" else 0, speed_of_impact]])
    prediction = model.predict(input_data)
    if prediction == 1:
        st.write("You are most likely to survive")
    else:
        st.write("You are most likely not going to survive")

