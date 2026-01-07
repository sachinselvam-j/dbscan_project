import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="DBSCAN Clustering App", layout="centered")

st.title("DBSCAN Clustering App")

# Load saved model and scaler
with open("dbscan_model.pkl", "rb") as f:
    dbscan_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features
    features = st.multiselect(
        "Select exactly TWO features for clustering",
        df.columns,
        default=['Annual Income (k$)', 'Spending Score (1-100)']
    )

    if len(features) == 2:
        X = df[features]

        # Scale input
        X_scaled = scaler.fit_transform(X)

        # DBSCAN clustering (re-fit is REQUIRED)
        clusters = dbscan_model.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.write(df.head())

        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=clusters
        )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("DBSCAN Clustering Result")

        st.pyplot(fig)

        # Cluster info
        st.subheader("Cluster Distribution")
        st.write(df["Cluster"].value_counts())

    else:
        st.warning("Please select exactly TWO features.")

