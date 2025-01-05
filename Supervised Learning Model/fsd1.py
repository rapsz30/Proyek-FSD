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
    nb_model = GaussianNB(priors=[0.6, 0.4])  # Adjust priors for better balance
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

# Helper function to safely convert string to int
def safe_int_convert(value, default=0):
    try:
        return int(value.split('-')[0])
    except (ValueError, AttributeError, IndexError):
        return default

# Calculate risk score
def calculate_risk_score(age, sex, family_history, diabetes, smoking, sbp, tch):
    risk_score = 0
    
    # Age risk
    age_start = safe_int_convert(age)
    if age_start >= 55:
        risk_score += 2
    elif age_start >= 45:
        risk_score += 1
    
    # Family history risk
    if family_history == "Yes":
        risk_score += 1
    
    # Diabetes risk
    if diabetes == "Yes":
        risk_score += 2
    
    # Smoking risk
    if smoking == "Current":
        risk_score += 2
    elif smoking == "Former":
        risk_score += 1
    
    # Blood pressure risk
    sbp_start = safe_int_convert(sbp)
    if sbp_start >= 140:
        risk_score += 2
    elif sbp_start >= 130:
        risk_score += 1
    
    # Cholesterol risk
    tch_start = safe_int_convert(tch)
    if tch_start >= 240:
        risk_score += 2
    elif tch_start >= 200:
        risk_score += 1
    
    # Adjust for sex
    if sex == "Female":
        risk_score *= 0.7  # Further reduce risk score for females
    
    return risk_score

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
        age = st.selectbox("Age", sorted(data['Age'].unique()))
        sex = st.selectbox("Sex", sorted(data['Sex'].unique()))
        family_history = st.selectbox("Family history of CVD", sorted(data['Family history of CVD'].unique()))
        diabetes = st.selectbox("Diabetes Mellitus", sorted(data['Diabetes Mellitus'].unique()))
        smoking = st.selectbox("Smoking status", sorted(data['Smoking status'].unique()))
        sbp = st.selectbox("SBP", sorted(data['SBP'].unique()))
        tch = st.selectbox("Tch", sorted(data['Tch'].unique()))

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

                # Encode the input data
                input_encoded = pd.DataFrame()
                for col in input_data.columns:
                    le = label_encoders[col]
                    input_encoded[col] = le.transform(input_data[col].astype(str))

                # Scale the encoded input
                input_scaled = scaler.transform(input_encoded)

                # Make prediction
                pred_proba = nb_model.predict_proba(input_scaled)[0]
                
                # Calculate risk score
                risk_score = calculate_risk_score(age, sex, family_history, diabetes, smoking, sbp, tch)
                
                # Adjust final probability based on risk score and sex
                if sex == "Female":
                    base_prob = pred_proba[1] * 0.75  # Further reduce base probability for females
                    final_prob = (base_prob * 0.7 + (risk_score / 20) * 0.3)  # Weighted average
                else:
                    final_prob = (pred_proba[1] * 0.8 + (risk_score / 15) * 0.2)  # Weighted average

                final_prob = max(0, min(1, final_prob))  # Ensure probability is between 0 and 1

                # Display results
                st.subheader("Hasil Prediksi")

                # Define risk levels
                if final_prob < 0.2:
                    risk_level = "Sangat Rendah"
                    risk_color = "green"
                elif final_prob < 0.4:
                    risk_level = "Rendah"
                    risk_color = "lightgreen"
                elif final_prob < 0.6:
                    risk_level = "Sedang"
                    risk_color = "yellow"
                elif final_prob < 0.8:
                    risk_level = "Tinggi"
                    risk_color = "orange"
                else:
                    risk_level = "Sangat Tinggi"
                    risk_color = "red"

                st.markdown(f"<h4 style='color: {risk_color}'>Tingkat Risiko: {risk_level}</h4>", 
                            unsafe_allow_html=True)

                st.write(f"Probabilitas Risiko: {final_prob:.2f}")

                # Display risk factors
                st.subheader("Faktor Risiko Teridentifikasi:")
                age_start = safe_int_convert(age)
                if age_start >= 55:
                    st.write("- Usia di atas 55 tahun")
                elif age_start >= 45:
                    st.write("- Usia di atas 45 tahun")
                if family_history == "Yes":
                    st.write("- Riwayat keluarga dengan CVD")
                if diabetes == "Yes":
                    st.write("- Diabetes Mellitus")
                if smoking == "Current":
                    st.write("- Perokok aktif")
                elif smoking == "Former":
                    st.write("- Mantan perokok")
                sbp_start = safe_int_convert(sbp)
                if sbp_start >= 140:
                    st.write("- Tekanan darah sistolik tinggi (≥140 mmHg)")
                elif sbp_start >= 130:
                    st.write("- Tekanan darah sistolik agak tinggi (130-139 mmHg)")
                tch_start = safe_int_convert(tch)
                if tch_start >= 240:
                    st.write("- Kolesterol total tinggi (≥240 mg/dL)")
                elif tch_start >= 200:
                    st.write("- Kolesterol total agak tinggi (200-239 mg/dL)")

            except Exception as e:
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
            - **Mohamad Rafi Hendryansah** (23523064)  
            - **Afifuddin Mahfud** (23523076)  
            - **Yusuf Aditya Kresnayana** (23523077)  
            - **Naufal Rizqy Wardono** (23523097)  
            - **Mustaqim Adiyatno** (23523107)  
            - **M. Trendo Rafly Dipu** (23523116)
            """
        )
        st.markdown("---")

        st.info("Jelajahi aplikasi ini untuk mempelajari lebih lanjut tentang kesehatan jantung Anda!")

if __name__ == "__main__":
    main()

