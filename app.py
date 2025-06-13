import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detector.pkl")

st.title("HireSafe 🛡️ - Job Post Fraud Detector")

uploaded_file = st.file_uploader("Upload a CSV file with job postings", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Combine text columns (match with training format)
    df['text'] = df['title'].fillna('') + ' ' + df['company_profile'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna('')
    
    X = df['text']
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df['Fraud Probability'] = probabilities
    df['Prediction'] = predictions
    df['Prediction Label'] = df['Prediction'].apply(lambda x: "Fraudulent" if x == 1 else "Genuine")
    
    st.write("### Prediction Results", df[['title', 'Prediction Label', 'Fraud Probability']])
    
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
