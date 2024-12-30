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
    # Try different delimiters
    delimiters = [',', ';', '\t']
    data = None
    
    for delimiter in delimiters:
        try:
            data = pd.read_csv("riskchartsampledata.csv", delimiter=delimiter)
            if len(data.columns) > 1:
                break
        except:
            continue
    
    if data is None or len(data.columns) == 1:
        st.error("Unable to read the CSV file correctly. Please check the file format.")
        st.stop()
    
    # Clean column names
    data.columns = data.columns.str.strip()
    
    # Remove any rows that are completely empty
    data = data.dropna(how='all')
    
    # Remove any columns that are completely empty
    data = data.dropna(axis=1, how='all')
    
    # Define the expected column names
    expected_columns = ['Age', 'Sex', 'Family history of CVD', 'Diabetes Mellitus', 'High WHR', 'Smoking status', 'SBP', 'Tch']
    
    # Check which columns are actually present in the data
    present_columns = [col for col in expected_columns if col in data.columns]
    
    # Encode categorical variables and handle numeric variables
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            # For categorical variables, use label encoding
            data[col] = le.fit_transform(data[col].astype(str))
        else:
            # For numeric variables, try to convert to float
            try:
                data[col] = data[col].astype(float)
            except ValueError:
                # If conversion fails, treat as categorical and use label encoding
                data[col] = le.fit_transform(data[col].astype(str))
    
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

# The rest of the code remains the same

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
    data, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_data()
    lr_model, nb_model, svm_model = train_models(X_train_scaled, y_train)
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Evaluasi", "Prediksi", "Tentang Kami"])
    
    if page == "Dataset":
        st.header("Dataset")
        st.write(data)
        
        st.subheader("Visualisasi Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=data.columns[-1], data=data)
        plt.title(f"Distribusi {data.columns[-1]}")
        st.pyplot(fig)
        
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
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
        
        # Get the actual column names from the data
        columns = data.columns.tolist()
        columns.remove(data.columns[-1])  # Remove the target variable
        
        # Create input fields dynamically based on available columns
        input_data = {}
        for col in columns:
            unique_values = data[col].unique().tolist()
            input_data[col] = st.selectbox(col, unique_values)
        
        if st.button("Prediksi"):
            le = LabelEncoder()
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.columns:
                if data[col].dtype == 'object':
                    input_df[col] = le.fit_transform(input_df[col].astype(str))
                else:
                    input_df[col] = input_df[col].astype(float)
            
            input_scaled = scaler.transform(input_df)
            
            lr_pred = lr_model.predict_proba(input_scaled)[0][1]
            nb_pred = nb_model.predict_proba(input_scaled)[0][1]
            svm_pred = svm_model.predict_proba(input_scaled)[0][1]
            
            st.subheader("Hasil Prediksi")
            st.write(f"Regresi Logistik: {lr_pred:.2%} risiko {data.columns[-1]}")
            st.write(f"Naive Bayes: {nb_pred:.2%} risiko {data.columns[-1]}")
            st.write(f"Support Vector Machine: {svm_pred:.2%} risiko {data.columns[-1]}")
    
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