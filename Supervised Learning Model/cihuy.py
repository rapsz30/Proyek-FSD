import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Risk scoring function
def calculate_risk_score(user_data):
    score = 0
    
    # Age risk (higher risk for older age groups)
    age_scores = {
        '35-44': 0,
        '45-54': 1,
        '55-64': 2,
        '65-74': 3,
        '>=75': 4
    }
    score += age_scores.get(user_data.get('Age'), 0)
    
    # Family history risk
    if user_data.get('Family history of CVD') == 'Yes':
        score += 2
    
    # Diabetes risk
    if user_data.get('Diabetes Mellitus') == 'Diabetes':
        score += 2
    
    # Smoking risk
    if user_data.get('Smoking status') == 'Smoker':
        score += 2
    
    # Blood pressure risk
    sbp_scores = {
        '<120 mmHg': 0,
        '120-139 mmHg': 1,
        '140-159 mmHg': 2,
        '>=160 mmHg': 3
    }
    score += sbp_scores.get(user_data.get('SBP'), 0)
    
    # Cholesterol risk
    tch_scores = {
        '<150 mg/dL': 0,
        '150-200 mg/dL': 1,
        '200-250 mg/dL': 2,
        '250-300 mg/dL': 3,
        '>=300 mg/dL': 4
    }
    score += tch_scores.get(user_data.get('Tch'), 0)
    
    # High WHR risk
    if user_data.get('High WHR') == 'Yes':
        score += 1
    
    return score

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(data, target_column):
    # Use the first 100 rows
    data = data.head(100)
    
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Encode categorical variables
    encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        encoders[column] = le

    return X, y, data, encoders

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return model, accuracy, precision, recall, f1

# Streamlit app
def main():
    st.title("Heart Disease Risk Prediction")
    st.write("This app predicts the risk of heart disease based on various health factors.")

    # File uploader
    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV file
        data = pd.read_csv(uploaded_file, sep=';')
        
        # Display the first few rows of the data
        st.subheader("Data Preview (First 100 rows)")
        st.write(data.head(100))

        # Allow user to select the target column
        target_column = st.selectbox("Select the target column (Risk)", data.columns)

        if st.button("Train Model"):
            # Preprocess data
            X, y, processed_data, encoders = load_and_preprocess_data(data, target_column)

            # Train model
            model, accuracy, precision, recall, f1 = train_model(X, y)

            # Store model and data in session state
            st.session_state['model'] = model
            st.session_state['processed_data'] = processed_data
            st.session_state['feature_columns'] = X.columns
            st.session_state['encoders'] = encoders

            # Display model performance
            st.subheader("Model Performance")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-score: {f1:.2f}")

    # Check if model has been trained
    if 'model' in st.session_state:
        st.subheader("Enter Your Information")
        user_input = {}
        for column in st.session_state['feature_columns']:
            unique_values = sorted(st.session_state['processed_data'][column].unique())
            user_input[column] = st.selectbox(f"{column}", unique_values)

        # Make prediction
        if st.button("Predict Risk"):
            # Calculate risk score
            risk_score = calculate_risk_score(user_input)
            
            # Prepare data for model prediction
            user_data = pd.DataFrame(user_input, index=[0])
            
            # Encode user input
            for column in user_data.columns:
                le = st.session_state['encoders'][column]
                user_data[column] = le.transform(user_data[column])

            # Make prediction using model probability
            prediction_prob = st.session_state['model'].predict_proba(user_data)[0]
            
            # Calculate maximum possible risk score
            max_risk_score = 16  # Sum of maximum possible points
            
            # Normalize risk score to 0-1 range
            normalized_risk_score = risk_score / max_risk_score
            
            # Combine model probability and risk score
            final_risk_prob = (prediction_prob[1] + normalized_risk_score) / 2
            
            # Determine risk level based on combined probability
            risk_level = "High" if final_risk_prob > 0.4 else "Low"

            st.subheader("Prediction Result")
            st.write(f"The predicted risk of heart disease is: {risk_level}")
            
            # Display risk factors explanation
            st.subheader("Risk Analysis")
            st.write(f"Risk Score: {risk_score} out of {max_risk_score}")
            st.write(f"Combined Risk Probability: {final_risk_prob:.2f}")
            st.write("\nRisk Factors Found:")
            
            if user_input['Age'] in ['55-64', '65-74', '>=75']:
                st.write("- Advanced age increases risk")
            if user_input['Family history of CVD'] == 'Yes':
                st.write("- Family history of cardiovascular disease")
            if user_input['Diabetes Mellitus'] == 'Diabetes':
                st.write("- Presence of diabetes")
            if user_input['Smoking status'] == 'Smoker':
                st.write("- Active smoking status")
            if user_input['SBP'] in ['>=160 mmHg', '140-159 mmHg']:
                st.write("- Elevated blood pressure")
            if user_input['Tch'] in ['>=300 mg/dL', '250-300 mg/dL', '200-250 mg/dL']:
                st.write("- High cholesterol levels")
            if user_input['High WHR'] == 'Yes':
                st.write("- High waist-to-hip ratio")

            # Display recommendation based on risk level
            st.subheader("Recommendations")
            if risk_level == "High":
                st.write("""
                Based on your risk factors, it is recommended to:
                1. Consult with a healthcare provider
                2. Monitor blood pressure regularly
                3. Maintain a healthy diet
                4. Exercise regularly
                5. Consider smoking cessation if applicable
                6. Monitor blood sugar levels if diabetic
                """)
            else:
                st.write("""
                While your risk level is currently low, it's important to:
                1. Maintain a healthy lifestyle
                2. Have regular health check-ups
                3. Stay physically active
                4. Maintain a balanced diet
                """)

    else:
        st.info("Please upload a CSV file and train the model to proceed.")

if __name__ == "__main__":
    main()

