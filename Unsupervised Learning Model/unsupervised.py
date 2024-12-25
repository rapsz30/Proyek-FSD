import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn.impute import SimpleImputer

# Function to load data
@st.cache_data
def load_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

# Function to preprocess data
def preprocess_data(df):
    # Convert data types
    df['Tempat Tinggal'] = df['Tempat Tinggal'].astype(int)
    df['Jumlah Tanggungan Orang Tua'] = df['Jumlah Tanggungan Orang Tua'].astype(int)
    df['Kendaraan'] = df['Kendaraan'].astype(int)
    df['Kelayakan Keringanan UKT'] = df['Kelayakan Keringanan UKT'].astype(int)

    # Convert categorical variables to numerical
    df['Pekerjaan Orang Tua'] = pd.Categorical(df['Pekerjaan Orang Tua']).codes
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Normalize numerical features
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
    
    return normalized_df

# Function to perform K-means clustering
def perform_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    return clusters, kmeans

# Function to visualize clusters
def visualize_clusters(df, clusters, x_col, y_col):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df[x_col], df[y_col], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'K-means Clustering: {x_col} vs {y_col}')
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title('Unsupervised Learning Model:')

    try:
        # Load data
        file_path = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/klasifikasimhs-xwuq9Zr5EncwomoEdAoDI2H4qkHWHA.csv"
        df = load_data(file_path)

        # Display raw data
        st.subheader("Raw Data")
        st.write(df)

        # Preprocess data
        preprocessed_df = preprocess_data(df)

        # Perform K-means clustering
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        clusters, kmeans = perform_kmeans(preprocessed_df, n_clusters)

        # Add cluster labels to the original dataframe
        df['Cluster'] = clusters

        # Display cluster statistics
        st.subheader("Cluster Statistics")
        for i in range(n_clusters):
            st.write(f"Cluster {i}:")
            st.write(df[df['Cluster'] == i].describe())

        # Visualize clusters
        st.subheader("Cluster Visualization")
        x_col = st.selectbox("Select X-axis feature", df.columns[:-2])
        y_col = st.selectbox("Select Y-axis feature", df.columns[:-2])
        visualize_clusters(df, clusters, x_col, y_col)

        # Analyze cluster characteristics
        st.subheader("Cluster Characteristics")
        for i in range(n_clusters):
            st.write(f"Cluster {i}:")
            cluster_data = df[df['Cluster'] == i]
            st.write(f"Average Penghasilan Orang Tua: {cluster_data['Penghasilan Orang Tua'].mean():.2f}")
            st.write(f"Average Jumlah Tanggungan Orang Tua: {cluster_data['Jumlah Tanggungan Orang Tua'].mean():.2f}")
            st.write(f"Most common Pekerjaan Orang Tua: {cluster_data['Pekerjaan Orang Tua'].mode().values[0]}")
            st.write(f"Percentage of students eligible for financial aid: {(cluster_data['Kelayakan Keringanan UKT'] == 0).mean() * 100:.2f}%")

        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = df.drop(['Cluster', 'Kelayakan Keringanan UKT'], axis=1).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

