import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.impute import SimpleImputer

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv("riskchartsampledata.csv")
    
   # Define the expected column names
    expected_columns = ['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 'Smoking status', 'SBP', 'Tch']
    
    # Check which columns are actually present in the data
    present_columns = [col for col in expected_columns if col in data.columns]
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in present_columns:
        data[col] = le.fit_transform(data[col])
    
    # Ensure 'High WHR' is present, otherwise use the last column as the target
    if 'High WHR' in data.columns:
        target_column = 'High WHR'
    else:
        target_column = data.columns[-1]
        st.warning(f"'High WHR' column not found. Using '{target_column}' as the target variable.")
    
    # Split features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return data, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# Train models
@st.cache_resource
def train_models(X_train_scaled, y_train):
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    return lr_model, nb_model, svm_model

# Evaluate models
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return cm, cr, mse, rmse

# Main Streamlit app
def main():
    st.title("Cardiovascular Disease Risk Prediction")
    
    # Load data and train models
    data, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_data()
    lr_model, nb_model, svm_model = train_models(X_train_scaled, y_train)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset", "Evaluation", "Prediction", "About"])
    
    if page == "Dataset":
        st.header("Dataset")
        st.write(data)
        
        st.subheader("Data Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=data.columns[-1], data=data)
        plt.title(f"Distribution of {data.columns[-1]}")
        st.pyplot(fig)
        
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        
    elif page == "Evaluation":
        st.header("Model Evaluation")
        
        model_option = st.selectbox("Select Model", ["Logistic Regression", "Naive Bayes", "SVM"])
        
        if model_option == "Logistic Regression":
            model = lr_model
        elif model_option == "Naive Bayes":
            model = nb_model
        else:
            model = svm_model
        
        cm, cr, mse, rmse = evaluate_model(model, X_test_scaled, y_test)
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - {model_option}")
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.write(pd.DataFrame(cr).transpose())
        
        st.subheader("Regression Metrics")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
    elif page == "Prediction":
        st.header("Predict Cardiovascular Disease Risk")
        
        age = st.selectbox("Age", ["35-44", "45-54", "55-64", "65-74", ">=75"])
        sex = st.selectbox("Sex", ["male", "female"])
        family_history = st.selectbox("Family history of CVD", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes Mellitus", ["Diabetes", "Non-diabetes"])
        smoking = st.selectbox("Smoking status", ["Smoker", "Non-smoker"])
        sbp = st.selectbox("Systolic Blood Pressure (SBP)", ["<120 mmHg", "120-139 mmHg", "140-159 mmHg", ">=160 mmHg"])
        tch = st.selectbox("Total Cholesterol (Tch)", ["<150 mg/dL", "150-200 mg/dL", "200-250 mg/dL", "250-300 mg/dL", ">=300 mg/dL"])
        
        if st.button("Predict"):
            le = LabelEncoder()
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'Family history of CVD': [family_history],
                'Diabetes Mellitus': [diabetes],
                'Smoking status': [smoking],
                'SBP': [sbp],
                'Tch': [tch]
            })
            
            for col in input_data.columns:
                input_data[col] = le.fit_transform(input_data[col])
            
            input_scaled = scaler.transform(input_data)
            
            lr_pred = lr_model.predict_proba(input_scaled)[0][1]
            nb_pred = nb_model.predict_proba(input_scaled)[0][1]
            svm_pred = svm_model.predict_proba(input_scaled)[0][1]
            
            st.subheader("Prediction Results")
            st.write(f"Logistic Regression: {lr_pred:.2%} risk of high WHR")
            st.write(f"Naive Bayes: {nb_pred:.2%} risk of high WHR")
            st.write(f"Support Vector Machine: {svm_pred:.2%} risk of high WHR")
            
    else:
        st.header("About Us")
        st.write("This application predicts the risk of cardiovascular disease based on various factors.")
        st.write("Created by: Your Name")
        st.write("Date: June 2023")

if __name__ == "__main__":
    main()