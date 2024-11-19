import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to train the model
@st.cache_resource
def train_model_and_encoders():
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

    # Encode categorical variables with separate encoders
    encoders = {}
    for col in ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area']:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    # Target variable encoding
    target_encoder = LabelEncoder()
    df['Loan_Status'] = target_encoder.fit_transform(df['Loan_Status'])
    encoders['Loan_Status'] = target_encoder

    # Split the data
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    return model, encoders

# Load trained model and encoders
model, encoders = train_model_and_encoders()

# Function to encode inputs with fallback for unseen labels
def encode_input(value, encoder, default_value, column_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        st.warning(f"Unrecognized value for '{column_name}': {value}. Using default '{default_value}'.")
        return encoder.transform([default_value])[0]

# Streamlit App
def run():
    st.title("Loan Prediction System")

    # Input Fields
    fn = st.text_input('Full Name')  # User's name

    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    married = st.selectbox("Married", encoders['Married'].classes_)
    dependents = st.selectbox("Dependents", encoders['Dependents'].classes_)
    education = st.selectbox("Education", encoders['Education'].classes_)
    self_employed = st.selectbox("Self Employed", encoders['Self_Employed'].classes_)

    applicant_income = st.number_input("Applicant's Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant's Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

    loan_term = st.number_input("Loan Term (Months)", min_value=0)
    credit_history = st.selectbox("Credit History", ['No History (0)', 'Good History (1)'])
    property_area = st.selectbox("Property Area", encoders['Property_Area'].classes_)

    # Prediction
    if st.button("Predict Loan Approval"):
        # Prepare input for the model
        input_data = {
            'Gender': encode_input(gender, encoders['Gender'], 'Male', 'Gender'),
            'Married': encode_input(married, encoders['Married'], 'No', 'Married'),
            'Dependents': encode_input(dependents, encoders['Dependents'], '0', 'Dependents'),
            'Education': encode_input(education, encoders['Education'], 'Graduate', 'Education'),
            'Self_Employed': encode_input(self_employed, encoders['Self_Employed'], 'No', 'Self Employed'),
            'LoanAmountLog': np.log(loan_amount + 1) if loan_amount > 0 else 0,
            'Loan_Amount_Term': loan_term,
            'Credit_History': 1 if credit_history == 'Good History (1)' else 0,
            'Property_Area': encode_input(property_area, encoders['Property_Area'], 'Urban', 'Property Area')
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure input_df columns match model features
        expected_columns = list(model.feature_names_in_)
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Make Prediction
        try:
            prediction = model.predict(input_df)
            result = 'Approved' if prediction[0] == 1 else 'Not Approved'

            # Display Result
            if prediction[0] == 1:
                st.success(f"Congratulations {fn}, your loan is {result}!")
            else:
                st.error(f"Sorry {fn}, your loan is {result}!")
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")

run()
