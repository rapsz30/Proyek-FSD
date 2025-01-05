import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

# Load and preprocess the data
@st.cache_data
def load_data():
    # Read the CSV file
    data = pd.read_csv("riskchartsampledata.csv", header=None)
    
    # Set the initial column name
    data.columns = ['combined']
    
    # Split the combined column into separate columns
    data[['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 
          'Smoking status', 'SBP', 'Tch']] = data['combined'].str.split(';', expand=True)
    
    # Drop the original combined column
    data = data.drop('combined', axis=1)
    
    # Drop the first row (index 0) which contains duplicate header
    data = data.iloc[1:]
    
    # Drop duplicate rows and reset index
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Remove any rows that are completely empty
    data = data.dropna(how='all')

    # Encode categorical variables for model training
    data_encoded = data.copy()
    label_encoders = {}
    
    # Create and store label encoders for each column
    for col in data_encoded.columns:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        label_encoders[col] = le

    # Split features and target
    X = data_encoded.drop('High WHR', axis=1)
    y = data_encoded['High WHR']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders

# Train Naive Bayes model
@st.cache_resource
def train_naive_bayes(X_train_scaled, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    return nb_model

# Evaluate model
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return cm, cr, mse, rmse

# Main Streamlit app
def main():
    st.title("Prediksi Risiko Penyakit Jantung menggunakan Naive Bayes")

    # Load data and train model
    data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders = load_data()
    nb_model = train_naive_bayes(X_train_scaled, y_train)

    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Evaluasi", "Prediksi", "Tentang Kami"])

    if page == "Dataset":
        st.header("Dataset")
        st.write(data)

        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title("Heatmap Korelasi")
        st.pyplot(fig)

    elif page == "Evaluasi":
        st.header("Evaluasi Model Naive Bayes")

        cm, cr, mse, rmse = evaluate_model(nb_model, X_test_scaled, y_test)

        st.subheader("Matriks Konfusi")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title("Matriks Konfusi - Naive Bayes")
        st.pyplot(fig)

        st.subheader("Laporan Klasifikasi")
        st.write(pd.DataFrame(cr).transpose())

        st.subheader("Metrik Evaluasi")
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
            try:
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'Family history of CVD': [family_history],
                    'Diabetes Mellitus': [diabetes],
                    'Smoking status': [smoking],
                    'SBP': [sbp],
                    'Tch': [tch]
                })

                # Encode the input data using the same label encoders
                input_encoded = pd.DataFrame()
                for col in input_data.columns:
                    le = label_encoders[col]
                    input_encoded[col] = le.transform(input_data[col].astype(str))

                # Scale the encoded input
                input_scaled = scaler.transform(input_encoded)

                # Make prediction
                pred_proba = nb_model.predict_proba(input_scaled)[0]
                pred_class = nb_model.predict(input_scaled)[0]
                
                # Get the original label for the prediction
                original_label = label_encoders['High WHR'].inverse_transform([pred_class])[0]

                # Display risk level with color                
                st.subheader("Hasil Prediksi")
                risk_color = "red" if pred_proba[1] > 0.5 else "green"
                risk_level = "Tinggi" if pred_proba[1] > 0.5 else "Rendah"
                st.markdown(f"<h4 style='color: {risk_color}'>Tingkat Risiko: {risk_level}</h4>", 
                          unsafe_allow_html=True)
                st.write(f"Probabilitas Risiko Rendah: {pred_proba[0]:.2f}")
                st.write(f"Probabilitas Risiko Tinggi: {pred_proba[1]:.2f}")
            except ValueError as e:
                st.error(f"Error dalam pemrosesan input: {str(e)}")
                st.error("Pastikan semua input valid dan sesuai format")

    else:
        # Header utama
        st.header("Tentang Kami")
        st.markdown(
            """
            Aplikasi ini dirancang untuk memprediksi risiko penyakit jantung menggunakan algoritma Naive Bayes. 
            Model ini mempertimbangkan berbagai faktor seperti usia, jenis kelamin, riwayat keluarga dengan 
            penyakit kardiovaskular (CVD), diabetes mellitus, rasio lingkar pinggang terhadap pinggul (WHR) 
            yang tinggi, status merokok, tekanan darah sistolik (SBP), dan kadar kolesterol total (Tch).
            """
        )

        # Subheader untuk tim
        st.subheader("Dibuat oleh Kelompok Sembarang Wes:")
        st.markdown(
            """
            - *Mohamad Rafi Hendryansah* (23523064)  
            - *Afifuddin Mahfud* (23523076)  
            - *Yusuf Aditya Kresnayana* (23523077)  
            - *Naufal Rizqy Wardono* (23523097)  
            - *Mustaqim Adiyatno* (23523107)  
            - *M. Trendo Rafly Dipu* (23523116)
            """
        )
        st.markdown("---")

        st.info("Jelajahi aplikasi ini untuk mempelajari lebih lanjut tentang kesehatan jantung Anda!")

if __name__ == "__main__":
    main()