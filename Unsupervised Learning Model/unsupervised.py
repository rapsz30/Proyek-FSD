import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sidebar Navigation
menu = st.sidebar.selectbox("Pilih Menu", ["Model", "Tentang Kami"])

if menu == "Model":
    # Title
    st.title("Unsupervised Learning with DBSCAN")

    # Load Dataset
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        # Load dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("Preview Dataset")
        st.dataframe(data.head())

        # Features Definition
        numerical_features = ['Penghasilan Orang Tua', 'Jumlah Tanggungan Orang Tua', 'Kendaraan']
        categorical_features = ['Tempat Tinggal', 'Pekerjaan Orang Tua']

        # Preprocessing
        st.subheader("Preprocessing")
        try:
            # Encode categorical data
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
            encoded_categorical = encoder.fit_transform(data[categorical_features])
            encoded_categorical_df = pd.DataFrame(
                encoded_categorical.toarray(), 
                columns=encoder.get_feature_names_out(categorical_features)
            )

            # Scale numerical data
            scaler = StandardScaler()
            scaled_numerical = scaler.fit_transform(data[numerical_features])
            scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features)

            # Combine all features
            processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

            st.write("Preprocessed Data:")
            st.dataframe(processed_data.head())

            # DBSCAN Parameters
            st.subheader("DBSCAN Parameters")
            eps = st.slider("Epsilon (eps)", 0.1, 10.0, step=0.1, value=0.5)
            min_samples = st.slider("Minimum Samples (min_samples)", 1, 20, step=1, value=5)

            # DBSCAN Clustering
            st.subheader("DBSCAN Clustering")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(processed_data)

            # Add clusters to the original data
            data['Cluster'] = clusters
            st.write("Clustered Data:")
            st.dataframe(data)

            # PCA for Visualization
            st.subheader("Cluster Visualization")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(processed_data)
            reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
            reduced_df['Cluster'] = clusters

            # Plot Clusters
            plt.figure(figsize=(10, 6))
            for cluster in reduced_df['Cluster'].unique():
                cluster_data = reduced_df[reduced_df['Cluster'] == cluster]
                plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
            plt.legend()
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.title('DBSCAN Cluster Visualization')
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    # Tentang Kami Page
    st.header("Tentang Kami")
    st.markdown(
        """
        Aplikasi ini dirancang untuk mempermudah eksplorasi algoritma DBSCAN dalam pembelajaran tanpa pengawasan.  
        Anda dapat mengunggah dataset Anda, menyesuaikan parameter, dan melihat visualisasi cluster yang dihasilkan.
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
