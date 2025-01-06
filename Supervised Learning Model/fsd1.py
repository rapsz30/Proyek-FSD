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
    data = pd.read_csv("riskchartsampledata.csv", header=None)
    data.columns = ['combined']
    data[['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 
          'Smoking status', 'SBP', 'Tch']] = data['combined'].str.split(';', expand=True)
    data = data.drop('combined', axis=1)
    data = data.iloc[1:].drop_duplicates().reset_index(drop=True).dropna(how='all')

    data_encoded = data.copy()
    label_encoders = {}
    for col in data_encoded.columns:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        label_encoders[col] = le

    X = data_encoded.drop('High WHR', axis=1)
    y = data_encoded['High WHR']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=data_encoded[['Sex', 'High WHR']]
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders

@st.cache_resource
def train_naive_bayes(X_train_scaled, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    return nb_model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return cm, cr, mse, rmse

def calculate_risk_level(prob_high_whr, sex, age_range, family_history, diabetes, smoking):
    if sex == 'Female':
        base_threshold = 0.4
    else:
        base_threshold = 0.5

    # Extract the average age from the age range
    age_start, age_end = map(int, age_range.split('-'))
    avg_age = (age_start + age_end) / 2

    # Adjust threshold based on age
    age_factor = min(avg_age / 100, 1)
    threshold = base_threshold * (1 + age_factor)

    # Adjust threshold based on other risk factors
    if family_history == 'Yes':
        threshold *= 0.9
    if diabetes == 'Yes':
        threshold *= 0.9
    if smoking == 'Current':
        threshold *= 0.9

    if prob_high_whr < threshold * 0.6:
        return "Sangat Rendah", "green"
    elif prob_high_whr < threshold * 0.8:
        return "Rendah", "lightgreen"
    elif prob_high_whr < threshold:
        return "Sedang", "yellow"
    elif prob_high_whr < threshold * 1.2:
        return "Tinggi", "orange"
    else:
        return "Sangat Tinggi", "red"

def main():
    st.title("Prediksi Risiko Penyakit Jantung menggunakan Naive Bayes")

    data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders = load_data()
    nb_model = train_naive_bayes(X_train_scaled, y_train)

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
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'Family history of CVD': [family_history],
                    'Diabetes Mellitus': [diabetes],
                    'Smoking status': [smoking],
                    'SBP': [sbp],
                    'Tch': [tch]
                })

                encoded_input = input_data.copy()
                for column in encoded_input.columns:
                    encoded_input[column] = label_encoders[column].transform(encoded_input[column])

                scaled_input = scaler.transform(encoded_input)
                prediction = nb_model.predict(scaled_input)
                prediction_prob = nb_model.predict_proba(scaled_input)[0]

                st.subheader("Hasil Prediksi")

                prob_high_whr = prediction_prob[1]
                risk_level, risk_color = calculate_risk_level(prob_high_whr, sex, age, family_history, diabetes, smoking)

                st.markdown(f"<h4 style='color: {risk_color}'>Tingkat Risiko: {risk_level}</h4>", 
                          unsafe_allow_html=True)
                

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

