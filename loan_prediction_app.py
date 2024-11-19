import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to train the model
@st.cache_resource
def train_model():
    # Load the dataset
    df = pd.read_csv('loan.csv')

    # Handle missing values
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
    df['Total_Income_Log'] = np.log(df['Total_Income'] + 1)

    # Drop unnecessary columns
    cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'Loan_ID']
    df = df.drop(columns=cols_to_drop)

    # Encode categorical variables
    label_enc = LabelEncoder()
    cols_to_encode = ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in cols_to_encode:
        df[col] = label_enc.fit_transform(df[col])

    # Split the data
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    return model, label_enc

# Load trained model and label encoder
model, label_encoder = train_model()

# Streamlit App
def run():
    st.title("Loan Prediction System")

    # Input Fields
    fn = st.text_input('Full Name')  # User's name

    gender_display = ['Female', 'Male']
    gender = st.selectbox("Gender", gender_display)

    marital_display = ['No', 'Yes']
    married = st.selectbox("Married", marital_display)

    dependents_display = ['0', '1', '2', '3+']
    dependents = st.selectbox("Dependents", dependents_display)

    education_display = ['Not Graduate', 'Graduate']
    education = st.selectbox("Education", education_display)

    self_employed_display = ['No', 'Yes']
    self_employed = st.selectbox("Self Employed", self_employed_display)

    applicant_income = st.number_input("Applicant's Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant's Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

    loan_term_display = [12, 36, 60, 84, 120]
    loan_term = st.selectbox("Loan Term (Months)", loan_term_display)

    credit_history_display = ['No History (0)', 'Good History (1)']
    credit_history = st.selectbox("Credit History", credit_history_display)

    property_area_display = ['Rural', 'Semiurban', 'Urban']
    property_area = st.selectbox("Property Area", property_area_display)

    # Prediction
    if st.button("Predict Loan Approval"):
        # Prepare input for the model
        input_data = {
            'Gender': label_encoder.transform([gender])[0],
            'Married': label_encoder.transform([married])[0],
            'Dependents': label_encoder.transform([dependents])[0],
            'Education': label_encoder.transform([education])[0],
            'Self_Employed': label_encoder.transform([self_employed])[0],
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': np.log(loan_amount + 1),
            'Loan_Amount_Term': loan_term,
            'Credit_History': 1 if credit_history == 'Good History (1)' else 0,
            'Property_Area': label_encoder.transform([property_area])[0]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make Prediction
        prediction = model.predict(input_df)
        result = 'Approved' if prediction[0] == 1 else 'Not Approved'

        # Display Result
        if prediction[0] == 1:
            st.success(f"Congratulations {fn}, your loan is {result}!")
        else:
            st.error(f"Sorry {fn}, your loan is {result}!")

run()
