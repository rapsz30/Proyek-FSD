import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.impute import SimpleImputer

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv("riskchartsampledata.csv", delimiter=';')
    
    # Clean column names
    data.columns = data.columns.str.strip().str.split(';', expand=True)
    data.columns = ['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 'Smoking status', 'SBP', 'Tch']
    
    # Split the single column into multiple columns
    data = data.iloc[:, 0].str.split(';', expand=True)
    data.columns = ['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 'Smoking status', 'SBP', 'Tch']
    
    # Remove any rows that are completely empty
    data = data.dropna(how='all')
    
    # Create a copy of the data for encoding
    data_encoded = data.copy()
    
    # Encode categorical variables for model training
    le = LabelEncoder()
    for col in data_encoded.columns:
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
    
    # Split features and target
    X = data_encoded.drop('High WHR', axis=1)
    y = data_encoded['High WHR']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

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
    st.title("Prediksi Risiko Penyakit Jantung")
    
    # Load data and train models
    data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_data()
    lr_model, nb_model, svm_model = train_models(X_train_scaled, y_train)
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Evaluasi", "Prediksi", "Tentang Kami"])
    
    if page == "Dataset":
        st.header("Dataset")
        st.write(data)
        
        st.subheader("Visualisasi Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='High WHR', data=data)
        plt.title("Distribusi High WHR")
        st.pyplot(fig)
        
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title("Heatmap Korelasi")
        st.pyplot(fig)
        
    elif page == "Evaluasi":
        st.header("Evaluasi Model")
        
        model_option = st.selectbox("Pilih Model", ["Regresi Logistik", "Naive Bayes", "SVM"])
        
        if model_option == "Regresi Logistik":
            model = lr_model
        elif model_option == "Naive Bayes":
            model = nb_model
        else:
            model = svm_model
        
        cm, cr, mse, rmse = evaluate_model(model, X_test_scaled, y_test)
        
        st.subheader("Matriks Konfusi")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(f"Matriks Konfusi - {model_option}")
        st.pyplot(fig)
        
        st.subheader("Laporan Klasifikasi")
        st.write(pd.DataFrame(cr).transpose())
        
        st.subheader("Metrik Regresi")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
    elif page == "Prediksi":
        st.header("Prediksi Risiko Penyakit Jantung")
        
        # Create input fields for each feature
        age = st.selectbox("Age", data['Age'].unique())
        sex = st.selectbox("Sex", data['Sex'].unique())
        family_history = st.selectbox("Family history of CVD", data['Family history of CVD'].unique())
        diabetes = st.selectbox("Diabetes Mellitus", data['Diabetes Mellitus'].unique())
        smoking = st.selectbox("Smoking status", data['Smoking status'].unique())
        sbp = st.selectbox("SBP", data['SBP'].unique())
        tch = st.selectbox("Tch", data['Tch'].unique())
        
        if st.button("Prediksi"):
            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'Family history of CVD': [family_history],
                'Diabetes Mellitus': [diabetes],
                'Smoking status': [smoking],
                'SBP': [sbp],
                'Tch': [tch]
            })
            
            # One-hot encode the input data
            input_encoded = pd.get_dummies(input_data)
            
            # Ensure all columns from training are present in input data
            for col in X_train.columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training data
            input_encoded = input_encoded[X_train.columns]
            
            # Scale the input data
            input_scaled = scaler.transform(input_encoded)
            
            # Make predictions
            lr_pred = lr_model.predict_proba(input_scaled)[0][1]
            nb_pred = nb_model.predict_proba(input_scaled)[0][1]
            svm_pred = svm_model.predict_proba(input_scaled)[0][1]
            
            st.subheader("Hasil Prediksi")
            st.write(f"Regresi Logistik: {lr_pred:.2%} risiko High WHR")
            st.write(f"Naive Bayes: {nb_pred:.2%} risiko High WHR")
            st.write(f"Support Vector Machine: {svm_pred:.2%} risiko High WHR")
    
    else:
        st.header("Tentang Kami")
        st.write("Aplikasi ini memprediksi risiko penyakit jantung berdasarkan berbagai faktor.")
        st.write("Dibuat oleh: Kelompok Sembarang Wes")
        st.write("Mohamad Rafi Hendryansah (23523064)")
        st.write("Afifuddin Mahfud (23523076)")
        st.write("Yusuf Aditya Kresnayana(23523077)")
        st.write("Naufal Rizqy Wardono (23523097)")
        st.write("Mustaqim Adiyatno(23523107)")
        st.write("M. Trendo Rafly Dipu(23523116)")
        
if __name__ == "__main__":
    main()

