import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

# Initialize LabelEncoders
le_sex = LabelEncoder()
le_family_history = LabelEncoder()
le_diabetes = LabelEncoder()
le_smoking = LabelEncoder()

# Function to load and preprocess data from uploaded file
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')  
        df = df.dropna(subset=['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'Smoking status'])  
        
        def convert_age(age):
            if isinstance(age, str):
                if '-' in age:  
                    start, end = age.split('-')
                    return (int(start) + int(end)) / 2 
                elif '>' in age or '<' in age:  
                    return 75 
            return age 
        
        df['Age'] = df['Age'].apply(convert_age)
        
        # Mapping English labels to Indonesian
        sex_mapping = {'male': 'pria', 'female': 'wanita'}
        family_history_mapping = {'Yes': 'Ya', 'No': 'Tidak'}
        diabetes_mapping = {'Diabetes': 'Diabetes', 'Non-diabetes': 'Non-diabetes'}
        smoking_mapping = {'Smoker': 'Perokok', 'Non-smoker': 'Tidak Merokok'}

        df['Sex'] = df['Sex'].map(sex_mapping)
        df['Family history of CVD'] = df['Family history of CVD'].map(family_history_mapping)
        df['Diabetes Mellitus'] = df['Diabetes Mellitus'].map(diabetes_mapping)
        df['Smoking status'] = df['Smoking status'].map(smoking_mapping)
        
        # Encoding categorical columns
        df['Sex'] = le_sex.fit_transform(df['Sex'])  
        df['Family history of CVD'] = le_family_history.fit_transform(df['Family history of CVD']) 
        df['Diabetes Mellitus'] = le_diabetes.fit_transform(df['Diabetes Mellitus'])
        df['Smoking status'] = le_smoking.fit_transform(df['Smoking status'])
        
        # Features (X) and target (y)
        X = df[['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'Smoking status']]
        y = df['High WHR'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        return X, y
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        st.stop()

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_model = SVC(kernel='rbf', random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    return svm_model, scaler, X_test_scaled, y_test

# Function to evaluate the model
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return conf_matrix, class_report

# Function to create visualizations
def create_visualizations(model, X, conf_matrix):
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriks Konfusi')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Feature Importance
    try:
        importance = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=42)
        feature_importance = importance.importances_mean
        feature_names = X.columns

        print("Feature Importance Values:", feature_importance)  # Debugging

        if any(feature_importance > 0.01):  # Only plot significant features
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_names, y=feature_importance)
            plt.title('Feature Importance')
            plt.xlabel('Fitur')
            plt.ylabel('Kepentingan')
            plt.xticks(rotation=45)
            plt.savefig('feature_importance.png')
            plt.close()
        else:
            print("Feature Importance terlalu kecil, tidak dibuat.")
    except Exception as e:
        print(f"Error saat membuat Feature Importance: {e}")
        print(f"Working Directory: {os.getcwd()}")
        print("Daftar file di direktori:", os.listdir())

# Main function to run the Streamlit app
def main():
    st.title('Supervised learning model: Prediksi Risiko Penyakit Jantung')

    # Load and preprocess data
    file_path = 'D:/File/Kuliah/Semester 3/Fundamen Sains Data/Tugas Akhir/Supervised Learning Model/riskchartsampledata.xlsx'
    X, y = load_and_preprocess_data(file_path)

    # Train the model
    model, scaler, X_test_scaled, y_test = train_model(X, y)

    # Evaluate the model
    conf_matrix, class_report = evaluate_model(model, X_test_scaled, y_test)

    # Create visualizations
    create_visualizations(model, X, conf_matrix)

    # Save the model and scaler
    joblib.dump(model, 'svm_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Streamlit app
    st.subheader('Feature Importance')
    if os.path.exists('feature_importance.png'):
        st.image('feature_importance.png')
    else:
        st.error("File feature_importance.png tidak ditemukan. Pastikan fungsi visualisasi berjalan dengan benar.")

    st.subheader('Performa Model')
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png')
    else:
        st.error("File confusion_matrix.png tidak ditemukan. Pastikan fungsi visualisasi berjalan dengan benar.")

    st.subheader('Laporan Klasifikasi')
    st.text(class_report)

    st.subheader('Input Pengguna')
    age = st.number_input('Usia', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Jenis Kelamin', options=['pria', 'wanita'])
    family_history = st.selectbox('Riwayat Keluarga CVD', options=['Ya', 'Tidak'])
    diabetes = st.selectbox('Diabetes', options=['Diabetes', 'Non-diabetes'])
    smoking = st.selectbox('Merokok', options=['Perokok', 'Tidak Merokok'])

    # Encoding user input for prediction
    sex_encoded = le_sex.transform([sex])[0]
    family_history_encoded = le_family_history.transform([family_history])[0]
    diabetes_encoded = le_diabetes.transform([diabetes])[0]
    smoking_encoded = le_smoking.transform([smoking])[0]

    user_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_encoded],
        'Family history of CVD': [family_history_encoded],
        'Diabetes Mellitus': [diabetes_encoded],
        'Smoking status': [smoking_encoded]
    })

    user_data_scaled = scaler.transform(user_data)

    if st.button('Prediksi'):
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1]

        st.subheader('Hasil Prediksi')
        if prediction[0] == 1:
            st.warning(f'Risiko tinggi penyakit jantung. Probabilitas: {probability:.2f}')
        else:
            st.success(f'Risiko rendah penyakit jantung. Probabilitas: {probability:.2f}')

if __name__ == '__main__':
    main()
