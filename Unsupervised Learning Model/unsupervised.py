import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.sparse import issparse

st.set_page_config(page_title="Advanced Unsupervised Learning")

st.title("Advanced Unsupervised Learning dengan DBSCAN")
st.write("Aplikasi ini menggunakan DBSCAN untuk menganalisis dataset klasifikasi mahasiswa dengan fitur evaluasi dan visualisasi yang lengkap.")

uploaded_file = st.file_uploader("Upload dataset Anda (format CSV):", type="csv")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset yang diunggah:")
        st.dataframe(data)

        st.subheader("Informasi Dataset")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("Statistik Deskriptif:")
        st.write(data.describe())

        st.subheader("1. Preprocessing Data")

        all_features = data.columns.tolist()
        numerical_features = st.multiselect(
            "Pilih fitur numerik:",
            all_features,
            default=['Penghasilan Orang Tua', 'Jumlah Tanggungan Orang Tua', 'Kendaraan']
        )
        categorical_features = st.multiselect(
            "Pilih fitur kategorikal:",
            [col for col in all_features if col not in numerical_features],
            default=['Tempat Tinggal', 'Pekerjaan Orang Tua']
        )

        st.write("### Penanganan Missing Values")
        numeric_impute_strategy = st.selectbox(
            "Pilih strategi untuk mengisi missing values numerik:",
            ["mean", "median", "most_frequent"]
        )

        data[numerical_features] = data[numerical_features].fillna(
            data[numerical_features].agg(numeric_impute_strategy)
        )
        data[categorical_features] = data[categorical_features].fillna("missing")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features),
            ]
        )

        st.subheader("2. Parameter Tuning")
        col1, col2 = st.columns(2)
        
        with col1:
            eps_value = st.slider("Pilih nilai epsilon (eps):", 0.1, 10.0, 2.0, 0.1)
        with col2:
            min_samples_value = st.slider("Pilih jumlah minimum sampel (min_samples):", 1, 10, 5)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("dbscan", DBSCAN(eps=eps_value, min_samples=min_samples_value)),
        ])

        st.subheader("3. Clustering dengan DBSCAN")
        X_transformed = preprocessor.fit_transform(data)
        cluster_labels = pipeline.fit_predict(data)

        st.subheader("4. Evaluasi Model")
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        st.write(f"Jumlah cluster yang ditemukan: {n_clusters}")
        st.write(f"Jumlah noise points: {n_noise}")
        
        if n_clusters > 1:
            X_dense = X_transformed.toarray() if issparse(X_transformed) else X_transformed
            silhouette_avg = silhouette_score(X_dense, cluster_labels)
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")

        data['Cluster'] = cluster_labels
        st.write("Dataset dengan hasil clustering:")
        st.dataframe(data)

        st.subheader("5. Visualisasi Hasil")

        if X_transformed.shape[1] > 2:
            pca = PCA(n_components=2)
            X_dense = X_transformed.toarray() if issparse(X_transformed) else X_transformed
            X_pca = pca.fit_transform(X_dense)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title('PCA Visualization of Clusters')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            st.pyplot(fig)

        if len(numerical_features) >= 2:
            st.write("### Scatter Plot Matrix")
            fig = sns.pairplot(data, vars=numerical_features, hue='Cluster', palette='viridis')
            st.pyplot(fig)

        st.subheader("6. Analisis Cluster")
        for cluster in sorted(set(cluster_labels)):
            cluster_data = data[data['Cluster'] == cluster]
            st.write(f"### Cluster {cluster} ({'noise' if cluster == -1 else f'size: {len(cluster_data)}'})")
            st.write("Statistik cluster:")
            st.write(cluster_data[numerical_features].describe())

        st.subheader("7. Export Hasil")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download hasil clustering sebagai CSV",
            data=csv,
            file_name="hasil_clustering.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data: {str(e)}")

else:
    st.write("Silakan upload dataset Anda terlebih dahulu.")