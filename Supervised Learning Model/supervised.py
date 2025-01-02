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

# Load and preprocess the data
@st.cache_data
def load_data():
    # Read the CSV file
    data = pd.read_csv("riskchartsampledata.csv", header=None)

    # Drop duplicate rows (including header duplication)
    data = data.drop_duplicates()

    # Set the column names
    data.columns = ['combined']

    # Split the combined column into separate columns
    data[['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 'Smoking status', 'SBP', 'Tch']] = data['combined'].str.split(';', expand=True)

    # Drop the original combined column
    data = data.drop('combined', axis=1)

    # Remove any rows that are completely empty
    data = data.dropna(how='all')

    # Encode categorical variables for model training
    data_encoded = data.copy()
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

            # Encode and scale the input data
            input_encoded = pd.get_dummies(input_data)
            for col in X_train.columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[X_train.columns]
            input_scaled = scaler.transform(input_encoded)

            # Select a model and make predictions
            pred_model = st.selectbox("Pilih Model Prediksi", ["Regresi Logistik", "Naive Bayes", "SVM"])
            if pred_model == "Regresi Logistik":
                pred = lr_model.predict_proba(input_scaled)[0][1]
            elif pred_model == "Naive Bayes":
                pred = nb_model.predict_proba(input_scaled)[0][1]
            else:
                pred = svm_model.predict_proba(input_scaled)[0][1]

            st.subheader("Hasil Prediksi")
            st.write(f"Risiko {'Tinggi' if pred > 0.5 else 'Rendah'} penyakit jantung.")
            st.write(f"Probabilitas: {pred:.2f}")
    else:

# Header utama
        st.header("Tentang Kami")
        st.markdown(
            """
            Aplikasi ini dirancang untuk memprediksi risiko penyakit jantung berdasarkan berbagai faktor, seperti usia, jenis kelamin, riwayat keluarga dengan penyakit kardiovaskular (CVD), diabetes mellitus, rasio lingkar pinggang terhadap pinggul (WHR) yang tinggi, status merokok, tekanan darah sistolik (SBP), dan kadar kolesterol total (Tch). Kami berharap aplikasi ini dapat membantu pengguna untuk memahami risiko mereka dan mengambil langkah preventif yang tepat untuk menjaga kesehatan jantung
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