import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import math

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
def calculate_risk_score(age, sex, diabetes, smoking, sbp, tch):
    # Initialize variables
    age_score = 0
    diabetes_score = 0
    smoking_score = 0
    sbp_score = 0
    tch_score = 0
    
    # Calculate Age score
    age_start = safe_int_convert(age)
    if 35 <= age_start <= 44:
        age_value = math.log(40)
    elif 45 <= age_start <= 54:
        age_value = math.log(50)
    elif 55 <= age_start <= 64:
        age_value = math.log(60)
    elif 65 <= age_start <= 74:
        age_value = math.log(70)
    else:
        age_value = math.log(80)
    
    if sex == "Male":
        age_score = age_value * 3.06117
    elif sex == "Female":
        age_score = age_value * 2.32888
    
    # Calculate Diabetes score
    diabetes_value = 1 if diabetes == "Yes" else 0
    if sex == "Male":
        diabetes_score = diabetes_value * 0.57367
    elif sex == "Female":
        diabetes_score = diabetes_value * 0.69154

    # Calculate Smoking score
    smoking_value = 1 if smoking == "Current" else 0
    if sex == "Male":
        smoking_score = smoking_value * 0.65451
    elif sex == "Female":
        smoking_score = smoking_value * 0.52873
    
    # Calculate SBP score
    sbp_start = safe_int_convert(sbp)
    if sbp_start < 120:
        sbp_value = math.log(115)
    elif 120 <= sbp_start <= 139:
        sbp_value = math.log(130)
    elif 140 <= sbp_start <= 159:
        sbp_value = math.log(150)
    else:
        sbp_value = math.log(170)
    
    if sbp_start >= 160:
        if sex == "Male":
            sbp_score = sbp_value * 1.99881
        elif sex == "Female":
            sbp_score = sbp_value * 2.82263
    else:
        if sex == "Male":
            sbp_score = sbp_value * 1.93303
        elif sex == "Female":
            sbp_score = sbp_value * 2.76157
    
    # Calculate TCH score
    tch_start = safe_int_convert(tch)
    if tch_start < 150:
        tch_value = math.log(125)
    elif 150 <= tch_start <= 200:
        tch_value = math.log(175)
    elif 200 <= tch_start <= 250:
        tch_value = math.log(225)
    elif 250 <= tch_start <= 300:
        tch_value = math.log(275)
    else:
        tch_value = math.log(325)
    
    if sex == "Male":
        tch_score = tch_value * 1.1237
    elif sex == "Female":
        tch_score = tch_value * 1.20904
    
    # Calculate final risk score using the corrected formula
    x = age_score + diabetes_score + smoking_score + sbp_score + tch_score
    y = x - 23.9802
    
    # Langsung menggunakan rumus yang diberikan tanpa modifikasi
    risk_score = float(1 - (pow(0.88936, pow(math.e, y))))
    
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

    # Update the Prediksi section in main() function to use new risk calculation
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
                # Calculate risk score using the new formula
                final_prob = calculate_risk_score(age, sex, diabetes, smoking, sbp, tch)
                
                # Display results
                st.subheader("Hasil Prediksi")

                # Define risk levels
                if final_prob < 0.1:
                    risk_level = "Sangat Rendah"
                    risk_color = "green"
                elif final_prob < 0.2:
                    risk_level = "Rendah"
                    risk_color = "lightgreen"
                elif final_prob < 0.3:
                    risk_level = "Sedang"
                    risk_color = "yellow"
                elif final_prob < 0.4:
                    risk_level = "Tinggi"
                    risk_color = "orange"
                else:
                    risk_level = "Sangat Tinggi"
                    risk_color = "red"

                st.markdown(f"<h4 style='color: {risk_color}'>Tingkat Risiko: {risk_level}</h4>", 
                            unsafe_allow_html=True)

                st.write(f"Probabilitas Risiko: {final_prob:.4f}")

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
                sbp_start = safe_int_convert(sbp)
                if sbp_start >= 140:
                    st.write("- Tekanan darah sistolik tinggi (≥140 mmHg)")
                elif sbp_start >= 120:
                    st.write("- Tekanan darah sistolik agak tinggi (120-139 mmHg)")
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