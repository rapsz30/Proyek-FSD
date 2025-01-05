import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from collections import Counter

# Memuat dan memproses data
@st.cache_data
def muat_data():
    # Membaca file CSV
    data = pd.read_csv("riskchartsampledata.csv", header=None)
    
    # Menetapkan nama kolom awal
    data.columns = ['gabungan']
    
    # Memisahkan kolom gabungan menjadi kolom-kolom terpisah
    data[['Usia', 'Jenis Kelamin', 'Riwayat Keluarga CVD', 'Diabetes Mellitus', 'WHR Tinggi', 
        'Status Merokok', 'TDS', 'Kolesterol Total']] = data['gabungan'].str.split(';', expand=True)
    
    # Menghapus kolom gabungan asli
    data = data.drop('gabungan', axis=1)
    
    # Menghapus baris pertama (indeks 0) yang berisi header duplikat
    data = data.iloc[1:]
    
    # Menghapus baris duplikat dan mengatur ulang indeks
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Menghapus baris yang sepenuhnya kosong
    data = data.dropna(how='all')

    # Mengkodekan variabel kategorikal untuk pelatihan model
    data_encoded = data.copy()
    label_encoders = {}
    
    # Membuat dan menyimpan label encoder untuk setiap kolom
    for col in data_encoded.columns:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        label_encoders[col] = le

    # Memisahkan data berdasarkan jenis kelamin
    male_data = data_encoded[data_encoded['Jenis Kelamin'] == label_encoders['Jenis Kelamin'].transform(['male'])[0]]
    female_data = data_encoded[data_encoded['Jenis Kelamin'] == label_encoders['Jenis Kelamin'].transform(['female'])[0]]

    # Memisahkan fitur dan target untuk masing-masing gender
    X_male = male_data.drop('WHR Tinggi', axis=1)
    y_male = male_data['WHR Tinggi']
    X_female = female_data.drop('WHR Tinggi', axis=1)
    y_female = female_data['WHR Tinggi']

    # Membagi data training dan testing untuk masing-masing gender
    X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(
        X_male, y_male, test_size=0.2, random_state=42)
    X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(
        X_female, y_female, test_size=0.2, random_state=42)

    # Menggabungkan data training dan testing
    X_train = pd.concat([X_train_male, X_train_female])
    X_test = pd.concat([X_test_male, X_test_female])
    y_train = pd.concat([y_train_male, y_train_female])
    y_test = pd.concat([y_test_male, y_test_female])

    # Menskalakan fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders

# Melatih model Naive Bayes dengan pembobotan kelas
@st.cache_resource
def latih_naive_bayes(X_train_scaled, y_train):
    # Menghitung bobot kelas
    class_weights = dict(Counter(y_train))
    total_samples = len(y_train)
    for key in class_weights:
        class_weights[key] = total_samples / (len(class_weights) * class_weights[key])
    
    # Normalisasi bobot
    sum_weights = sum(class_weights.values())
    class_priors = [class_weights[0]/sum_weights, class_weights[1]/sum_weights]
    
    nb_model = GaussianNB(priors=class_priors)
    nb_model.fit(X_train_scaled, y_train)
    return nb_model

# Fungsi yang diperbarui untuk penyesuaian prediksi
# Menggunakan distribusi dataset wanita untuk membuat threshold dinamis

def sesuaikan_prediksi(pred_proba, jenis_kelamin, input_features, female_data):
    """
    Menyesuaikan probabilitas prediksi berdasarkan jenis kelamin
    dan faktor risiko yang relevan dalam dataset wanita.
    
    Parameters:
        pred_proba (list): Probabilitas risiko [rendah, tinggi]
        jenis_kelamin (str): Jenis kelamin ('male' atau 'female')
        input_features (pd.DataFrame): Input fitur yang diberikan
        female_data (pd.DataFrame): Dataset wanita
    
    Returns:
        list: Probabilitas yang disesuaikan
    """
    if jenis_kelamin == 'female':
        # Hitung distribusi risiko tinggi berdasarkan faktor risiko dalam dataset wanita
        risiko_tinggi = female_data[female_data['WHR Tinggi'] == 1]
        total_wanita = len(female_data)

        # Faktor risiko utama
        merokok_rendah = len(risiko_tinggi[(risiko_tinggi['Status Merokok'] == 'Non-smoker')]) / total_wanita
        riwayat_rendah = len(risiko_tinggi[(risiko_tinggi['Riwayat Keluarga CVD'] == 'No')]) / total_wanita
        diabetes_rendah = len(risiko_tinggi[(risiko_tinggi['Diabetes Mellitus'] == 'Non-diabetes')]) / total_wanita

        # Jika semua faktor risiko rendah, sesuaikan threshold
        if (input_features['Status Merokok'].iloc[0] == 'Non-smoker' and 
            input_features['Riwayat Keluarga CVD'].iloc[0] == 'No' and
            input_features['Diabetes Mellitus'].iloc[0] == 'Non-diabetes'):
            # Gunakan rata-rata distribusi untuk menyesuaikan probabilitas
            rata_rendah = (merokok_rendah + riwayat_rendah + diabetes_rendah) / 3
            adjusted_risk_low = max(0.5, rata_rendah)  # Probabilitas minimum 0.5 untuk risiko rendah
            adjusted_risk_high = 1 - adjusted_risk_low
            return [adjusted_risk_low, adjusted_risk_high]

    return pred_proba

# Mengevaluasi model
def evaluasi_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return cm, cr, mse, rmse

# Aplikasi Streamlit utama
def main():
    st.title("Prediksi Risiko Penyakit Jantung menggunakan Naive Bayes")

    # Memuat data dan melatih model
    data, data_encoded, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders = muat_data()
    nb_model = latih_naive_bayes(X_train_scaled, y_train)

    # Sidebar
    st.sidebar.title("Navigasi")
    halaman = st.sidebar.radio("Pilih Halaman", ["Dataset", "Evaluasi", "Prediksi", "Tentang Kami"])

    if halaman == "Dataset":
        st.header("Dataset")
        st.write(data)

        st.subheader("Distribusi Risiko berdasarkan Jenis Kelamin")
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_by_gender = pd.crosstab(data['Jenis Kelamin'], data['WHR Tinggi'])
        risk_by_gender.plot(kind='bar', ax=ax)
        plt.title("Distribusi Risiko berdasarkan Jenis Kelamin")
        plt.xlabel("Jenis Kelamin")
        plt.ylabel("Jumlah")
        st.pyplot(fig)

    elif halaman == "Evaluasi":
        st.header("Evaluasi Model Naive Bayes")
        
        cm, cr, mse, rmse = evaluasi_model(nb_model, X_test_scaled, y_test)

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

    elif halaman == "Prediksi":
        st.header("Prediksi Risiko Penyakit Jantung")

<<<<<<< HEAD
        # Membuat input fields untuk setiap fitur
        usia = st.selectbox("Usia", data['Usia'].unique())
        jenis_kelamin = st.selectbox("Jenis Kelamin", data['Jenis Kelamin'].unique())
        riwayat_keluarga = st.selectbox("Riwayat Keluarga CVD", data['Riwayat Keluarga CVD'].unique())
        diabetes = st.selectbox("Diabetes Mellitus", data['Diabetes Mellitus'].unique())
        merokok = st.selectbox("Status Merokok", data['Status Merokok'].unique())
        tds = st.selectbox("TDS", data['TDS'].unique())
        kolesterol = st.selectbox("Kolesterol Total", data['Kolesterol Total'].unique())
=======
        # Create input fields for each feature
<<<<<<< HEAD
        age = st.selectbox("Age", data['Age'].unique())
        sex = st.selectbox("Sex", data['Sex'].unique())
        family_history = st.selectbox("Family history of CVD", data['Family history of CVD'].unique())
        diabetes = st.selectbox("Diabetes Mellitus", data['Diabetes Mellitus'].unique())
        smoking = st.selectbox("Smoking status", data['Smoking status'].unique())
        sbp = st.selectbox("SBP", data['SBP'].unique())
        tch = st.selectbox("Tch", data['Tch'].unique())
=======
        age = st.selectbox("Age", sorted(data['Age'].unique()))
        sex = st.selectbox("Sex", sorted(data['Sex'].unique()))
        family_history = st.selectbox("Family history of CVD", sorted(data['Family history of CVD'].unique()))
        diabetes = st.selectbox("Diabetes Mellitus", sorted(data['Diabetes Mellitus'].unique()))
        smoking = st.selectbox("Smoking status", sorted(data['Smoking status'].unique()))
        sbp = st.selectbox("SBP", sorted(data['SBP'].unique()))
        tch = st.selectbox("Tch", sorted(data['Tch'].unique()))
>>>>>>> 52ab4bb42d45d10b00059bf195b87cc570cd7b6f
>>>>>>> 84297b32c5397b0c4f9beb78fced6af725cfbcb6

        if st.button("Prediksi"):
            try:
                # Membuat DataFrame input
                input_data = pd.DataFrame({
                    'Usia': [usia],
                    'Jenis Kelamin': [jenis_kelamin],
                    'Riwayat Keluarga CVD': [riwayat_keluarga],
                    'Diabetes Mellitus': [diabetes],
                    'Status Merokok': [merokok],
                    'TDS': [tds],
                    'Kolesterol Total': [kolesterol]
                })

<<<<<<< HEAD
                # Mengkodekan data input menggunakan label encoder yang sama
=======
<<<<<<< HEAD
                # Encode the input data using the same label encoders
=======
                # Encode the input data
>>>>>>> 52ab4bb42d45d10b00059bf195b87cc570cd7b6f
>>>>>>> 84297b32c5397b0c4f9beb78fced6af725cfbcb6
                input_encoded = pd.DataFrame()
                for col in input_data.columns:
                    le = label_encoders[col]
                    input_encoded[col] = le.transform(input_data[col].astype(str))

                # Menskalakan input yang telah dikodekan
                input_scaled = scaler.transform(input_encoded)

                # Membuat prediksi
                pred_proba = nb_model.predict_proba(input_scaled)[0]
<<<<<<< HEAD
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
=======
                
                # Menyesuaikan prediksi berdasarkan jenis kelamin dan faktor risiko
                pred_proba = sesuaikan_prediksi(pred_proba, jenis_kelamin, input_data, data[data['Jenis Kelamin'] == jenis_kelamin])
                
                # Menentukan tingkat risiko
                tingkat_risiko = "Tinggi" if pred_proba[1] > 0.5 else "Rendah"
                warna_risiko = "red" if pred_proba[1] > 0.5 else "green"

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi")
                st.markdown(f"<h4 style='color: {warna_risiko}'>Tingkat Risiko: {tingkat_risiko}</h4>", 
                        unsafe_allow_html=True)
                
                # Menampilkan probabilitas
                st.write(f"Probabilitas Risiko Rendah: {pred_proba[0]:.2f}")
                st.write(f"Probabilitas Risiko Tinggi: {pred_proba[1]:.2f}")

<<<<<<< HEAD
                # Menampilkan interpretasi
                st.subheader("Interpretasi Hasil")
                interpretasi = f"""
                Berdasarkan input yang diberikan:
                - Usia: {usia}
                - Jenis Kelamin: {jenis_kelamin}
                - Riwayat Keluarga CVD: {riwayat_keluarga}
                - Status Diabetes: {diabetes}
                - Status Merokok: {merokok}
                - Tekanan Darah Sistolik: {tds}
                - Kolesterol Total: {kolesterol}
                
                Model memprediksi risiko {tingkat_risiko.lower()} dengan tingkat kepercayaan {max(pred_proba):.2%}.
                """
                st.write(interpretasi)
=======
                # Display risk factors if risk is high
                if final_prob > 0.5:
                    st.subheader("Faktor Risiko Teridentifikasi:")
                    if int(age) > 50:
                        st.write("- Usia di atas 50 tahun")
                    if int(sbp) > 130:
                        st.write("- Tekanan darah sistolik tinggi")
                    if family_history == "Yes":
                        st.write("- Riwayat keluarga dengan CVD")
                    if diabetes == "Yes":
                        st.write("- Diabetes Mellitus")
                    if smoking == "Current":
                        st.write("- Perokok aktif")
>>>>>>> 52ab4bb42d45d10b00059bf195b87cc570cd7b6f
>>>>>>> 84297b32c5397b0c4f9beb78fced6af725cfbcb6

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
            yang tinggi, status merokok, tekanan darah sistolik (TDS), dan kadar kolesterol total.
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

if __name__== "__main__":
    main()
