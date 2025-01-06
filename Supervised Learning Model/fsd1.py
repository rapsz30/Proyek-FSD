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
    data = pd.read_csv("riskchartsampledata.csv", header=None)
    data.columns = ['combined']
    data[['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 
          'Smoking status', 'SBP', 'Tch']] = data['combined'].str.split(';', expand=True)
    data = data.drop('combined', axis=1)
    data = data.iloc[1:].drop_duplicates().reset_index(drop=True).dropna(how='all')

    # Separate data by gender
    male_data = data[data['Sex'] == 'Male']
    female_data = data[data['Sex'] == 'Female']

    # Process male data
    male_encoded = male_data.copy()
    male_label_encoders = {}
    for col in male_encoded.columns:
        le = LabelEncoder()
        male_encoded[col] = le.fit_transform(male_encoded[col].astype(str))
        male_label_encoders[col] = le

    # Process female data
    female_encoded = female_data.copy()
    female_label_encoders = {}
    for col in female_encoded.columns:
        le = LabelEncoder()
        female_encoded[col] = le.fit_transform(female_encoded[col].astype(str))
        female_label_encoders[col] = le

    # Prepare male training data
    X_male = male_encoded.drop(['High WHR', 'Sex'], axis=1)
    y_male = male_encoded['High WHR']

    # Prepare female training data
    X_female = female_encoded.drop(['High WHR', 'Sex'], axis=1)
    y_female = female_encoded['High WHR']

    # Split and scale male data
    X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(
        X_male, y_male, test_size=0.2, random_state=42
    )
    male_scaler = StandardScaler()
    X_male_train_scaled = male_scaler.fit_transform(X_male_train)
    X_male_test_scaled = male_scaler.transform(X_male_test)

    # Split and scale female data
    X_female_train, X_female_test, y_female_train, y_female_test = train_test_split(
        X_female, y_female, test_size=0.2, random_state=42
    )
    female_scaler = StandardScaler()
    X_female_train_scaled = female_scaler.fit_transform(X_female_train)
    X_female_test_scaled = female_scaler.transform(X_female_test)

    return (data, male_encoded, female_encoded, 
            X_male_train_scaled, X_male_test_scaled, y_male_train, y_male_test,
            X_female_train_scaled, X_female_test_scaled, y_female_train, y_female_test,
            male_scaler, female_scaler, male_label_encoders, female_label_encoders)

@st.cache_resource
def train_gender_specific_models(X_male_train_scaled, y_male_train, 
                               X_female_train_scaled, y_female_train):
    # Train male model
    male_model = GaussianNB()
    male_model.fit(X_male_train_scaled, y_male_train)
    
    # Train female model
    female_model = GaussianNB()
    female_model.fit(X_female_train_scaled, y_female_train)
    
    return male_model, female_model

def evaluate_models(male_model, female_model, 
                   X_male_test_scaled, y_male_test,
                   X_female_test_scaled, y_female_test):
    # Evaluate male model
    male_pred = male_model.predict(X_male_test_scaled)
    male_cm = confusion_matrix(y_male_test, male_pred)
    male_cr = classification_report(y_male_test, male_pred, output_dict=True)
    male_mse = mean_squared_error(y_male_test, male_pred)
    male_rmse = np.sqrt(male_mse)

    # Evaluate female model
    female_pred = female_model.predict(X_female_test_scaled)
    female_cm = confusion_matrix(y_female_test, female_pred)
    female_cr = classification_report(y_female_test, female_pred, output_dict=True)
    female_mse = mean_squared_error(y_female_test, female_pred)
    female_rmse = np.sqrt(female_mse)

    return (male_cm, male_cr, male_mse, male_rmse,
            female_cm, female_cr, female_mse, female_rmse)

def main():
    st.title("Prediksi Risiko Penyakit Jantung menggunakan Naive Bayes")

    # Load data and train models
    (data, male_encoded, female_encoded, 
     X_male_train_scaled, X_male_test_scaled, y_male_train, y_male_test,
     X_female_train_scaled, X_female_test_scaled, y_female_train, y_female_test,
     male_scaler, female_scaler, male_label_encoders, female_label_encoders) = load_data()

    male_model, female_model = train_gender_specific_models(
        X_male_train_scaled, y_male_train,
        X_female_train_scaled, y_female_train
    )

    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Evaluasi", "Prediksi", "Tentang Kami"])

    if page == "Dataset":
        st.header("Dataset")
        st.write(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Gender")
            gender_dist = data['Sex'].value_counts()
            fig, ax = plt.subplots()
            gender_dist.plot(kind='bar')
            plt.title("Distribusi Gender")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Distribusi WHR")
            whr_dist = data['High WHR'].value_counts()
            fig, ax = plt.subplots()
            whr_dist.plot(kind='bar')
            plt.title("Distribusi High WHR")
            st.pyplot(fig)

    elif page == "Evaluasi":
        st.header("Evaluasi Model Naive Bayes")

        (male_cm, male_cr, male_mse, male_rmse,
         female_cm, female_cr, female_mse, female_rmse) = evaluate_models(
            male_model, female_model,
            X_male_test_scaled, y_male_test,
            X_female_test_scaled, y_female_test
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Evaluasi Model Laki-laki")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(male_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title("Matriks Konfusi - Model Laki-laki")
            st.pyplot(fig)
            st.write(pd.DataFrame(male_cr).transpose())
            st.write(f"RMSE: {male_rmse:.4f}")

        with col2:
            st.subheader("Evaluasi Model Perempuan")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(female_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title("Matriks Konfusi - Model Perempuan")
            st.pyplot(fig)
            st.write(pd.DataFrame(female_cr).transpose())
            st.write(f"RMSE: {female_rmse:.4f}")

    elif page == "Prediksi":
        st.header("Prediksi Risiko Penyakit Jantung")

        col1, col2 = st.columns(2)
        
        with col1:
            age = st.selectbox("Age", sorted(data['Age'].unique()))
            sex = st.selectbox("Sex", sorted(data['Sex'].unique()))
            family_history = st.selectbox("Family history of CVD", sorted(data['Family history of CVD'].unique()))
            diabetes = st.selectbox("Diabetes Mellitus", sorted(data['Diabetes Mellitus'].unique()))

        with col2:
            smoking = st.selectbox("Smoking status", sorted(data['Smoking status'].unique()))
            sbp = st.selectbox("SBP", sorted(data['SBP'].unique()))
            tch = st.selectbox("Tch", sorted(data['Tch'].unique()))

        if st.button("Prediksi"):
            try:
                # Create input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Family history of CVD': [family_history],
                    'Diabetes Mellitus': [diabetes],
                    'Smoking status': [smoking],
                    'SBP': [sbp],
                    'Tch': [tch]
                })

                # Choose appropriate model and preprocessing based on gender
                if sex == 'Male':
                    label_encoders = male_label_encoders
                    scaler = male_scaler
                    model = male_model
                else:
                    label_encoders = female_label_encoders
                    scaler = female_scaler
                    model = female_model

                # Encode input
                encoded_input = input_data.copy()
                for column in encoded_input.columns:
                    encoded_input[column] = label_encoders[column].transform(encoded_input[column])

                # Scale input
                scaled_input = scaler.transform(encoded_input)

                # Make prediction
                prediction = model.predict(scaled_input)
                prediction_prob = model.predict_proba(scaled_input)[0]

                st.subheader("Hasil Prediksi")

                prob_high_whr = prediction_prob[1]

                if prob_high_whr < 0.2:
                    risk_level = "Sangat Rendah"
                    risk_color = "green"
                elif prob_high_whr < 0.4:
                    risk_level = "Rendah"
                    risk_color = "lightgreen"
                elif prob_high_whr < 0.6:
                    risk_level = "Sedang"
                    risk_color = "yellow"
                elif prob_high_whr < 0.8:
                    risk_level = "Tinggi"
                    risk_color = "orange"
                else:
                    risk_level = "Sangat Tinggi"
                    risk_color = "red"

                st.markdown(f"<h4 style='color: {risk_color}'>Tingkat Risiko: {risk_level}</h4>", 
                          unsafe_allow_html=True)
                
                st.write(f"Probabilitas Risiko WHR Tinggi: {prob_high_whr:.4f}")
                st.write(f"Probabilitas Risiko WHR Rendah: {prediction_prob[0]:.4f}")

                whr_status = "Tinggi" if prediction[0] == 1 else "Rendah"
                st.write(f"Prediksi WHR: {whr_status}")

            except Exception as e:
                st.error(f"Error dalam pemrosesan input: {str(e)}")
                st.error("Pastikan semua input valid dan sesuai format")

    else:
        st.header("Tentang Kami")
        st.markdown(
            """
            Aplikasi ini dirancang untuk memprediksi risiko penyakit jantung menggunakan algoritma Naive Bayes. 
            Model ini mempertimbangkan berbagai faktor seperti usia, jenis kelamin, riwayat keluarga dengan 
            penyakit kardiovaskular (CVD), diabetes mellitus, rasio lingkar pinggang terhadap pinggul (WHR) 
            yang tinggi, status merokok, tekanan darah sistolik (SBP), dan kadar kolesterol total (Tch).
            """
        )

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